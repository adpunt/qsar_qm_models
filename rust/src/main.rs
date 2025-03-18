use std::collections::HashMap;
use std::hash::Hash;
use regex::Regex;
use std::io::{self, BufReader, BufRead, Read, Seek, SeekFrom};
use ndarray::Array2;
use std::fs::File;
use serde::{Deserialize, Serialize};
use clap::{Arg, Command, ArgAction};
use num_traits::{Float, FromPrimitive};
use std::iter::Sum;
use rand_distr::{Distribution, Normal, Uniform, Beta, SkewNormal};
use std::io::Write;
use std::cmp::Reverse;
use std::io::BufWriter;
use std::fs::OpenOptions;

extern crate rdkit_sys;

use rdkit_sys::ro_mol_ffi::{smiles_to_mol};
use rdkit_sys::fingerprint_ffi::{rdk_fingerprint_mol, explicit_bit_vect_to_u64_vec}; // Assuming fingerprint generation is related to this type.
use cxx::let_cxx_string;
use cxx::UniquePtr;
use cxx::CxxVector;

const DELIMITER: u8 = 0x1F;  // ASCII 31 (Unit Separator)

struct SmilesTokenizer {
    regex: Regex,
}

impl SmilesTokenizer {
    fn new() -> Self {
        let regex_pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>>?|\*|\$|%[0-9]{2}|[0-9])";
        SmilesTokenizer {
            regex: Regex::new(regex_pattern).unwrap(),
        }
    }

    fn tokenize(&self, smiles: &str) -> Vec<String> {
        self.regex.find_iter(smiles).map(|mat| mat.as_str().to_owned()).collect()
    }
}

// TODO: get rust on ARC, right now I'm just copying the target file

#[derive(Deserialize, Debug)]
struct Config {
    sample_size: usize,
    noise: bool,
    train_count: usize, 
    test_count: usize,
    max_vocab: usize,
    iteration_seed: usize,
    molecular_representations: Vec<String>,
    k_domains: usize, 
    logging: bool,
    regression: bool,
}

#[derive(Debug)]
struct SmilesData {
    isomeric_smiles: String,
    canonical_smiles: String,
    randomized_smiles: Option<String>,
    target_value: f32,
    sns_buf: [u8; 16]
}

#[derive(Serialize, Clone)]
struct PlotPoint<T> {
    x: T,
    y: T,
}

#[derive(Debug, Clone)]
enum NoiseDistribution {
    Gaussian,
    LeftTailed,
    RightTailed,
    UShaped,
    Uniform,
    DomainMpnn,
    DomainTanimoto,
}

fn generate_noise_by_indices(
    indices: &[usize],
    mu: f32,
    sigma: f32,
    distribution: NoiseDistribution,
    alpha_skew: f32,
    beta_params: (f32, f32),
    seed: u64,
) -> HashMap<usize, f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut noise_map = HashMap::new();

    for &idx in indices {
        let noise = match distribution {
            NoiseDistribution::Gaussian | NoiseDistribution::DomainMpnn | NoiseDistribution::DomainTanimoto => {
                let normal = Normal::new(mu, sigma).unwrap();
                normal.sample(&mut rng)
            }
            NoiseDistribution::LeftTailed => {
                let skew_normal = SkewNormal::new(-alpha_skew, mu, sigma).unwrap();
                skew_normal.sample(&mut rng)
            }
            NoiseDistribution::RightTailed => {
                let skew_normal = SkewNormal::new(alpha_skew, mu, sigma).unwrap();
                skew_normal.sample(&mut rng)
            }
            NoiseDistribution::UShaped => {
                let beta = Beta::new(beta_params.0, beta_params.1).unwrap();
                mu + sigma * (beta.sample(&mut rng) - 0.5) * 2.0
            }
            NoiseDistribution::Uniform => {
                let uniform = Uniform::new_inclusive(mu - sigma, mu + sigma);
                uniform.sample(&mut rng)
            }
        };

        // Convert noise to f32 before inserting it into noise_map
        noise_map.insert(idx, noise as f32);
    }

    noise_map
}

// TODO: test will all molecular_representations, then delete extraneous print statements
// TODO: test the smiles that are failing, are they flawed or is my process not accepting certain strings?
fn read_smiles_data(
    reader: &mut BufReader<File>,
    molecular_representations: Vec<std::string::String>,
    k_domains: usize,
) -> Option<SmilesData> {
    let mut buffer = Vec::new();
    let mut delimiter_buf = [0u8; 1];

    let start_pos = reader.stream_position().unwrap();

    // Read isomeric_smiles and ensure it's in proper form
    buffer.clear();
    if reader.read_until(DELIMITER, &mut buffer).is_err() || buffer.is_empty() {
        return None; 
    }
    let isomeric_smiles_raw = &buffer[..buffer.len().saturating_sub(1)];
    let isomeric_smiles = match String::from_utf8(isomeric_smiles_raw.to_vec()) {
        Ok(s) if s.chars().all(|c| c.is_ascii_graphic() || c.is_whitespace()) && !s.is_empty() => {
            // Additional check: Ensure no weird characters (e.g., single quotes, unexpected punctuation)
            if s.chars().any(|c| c == '\u{FFFD}' || c == '\0' || c == '\'' || c == '�') {
                eprintln!("Skipping SMILES with bad characters: {:?}", s);
                return None; // Skip this entry
            }
            
            // Ensure SMILES length is within a reasonable range
            let smiles_len = s.len();
            if smiles_len < 5 || smiles_len > 300 {
                eprintln!("Skipping unusual SMILES length ({} chars): {:?}", smiles_len, s);
                return None;
            }

            s
        }
        _ => return None, // Invalid UTF-8 or empty string
    };

    // Read canonical_smiles
    buffer.clear();
    reader.read_until(DELIMITER, &mut buffer).ok()?;
    let canonical_smiles = String::from_utf8_lossy(&buffer[..buffer.len() - 1]).to_string();

    // Read property_value (float)
    let mut property_buf = [0u8; 4];
    reader.read_exact(&mut property_buf).ok()?;
    let target_value = f32::from_le_bytes(property_buf);
    reader.read_exact(&mut delimiter_buf).ok()?;

    // Read randomized_smiles (optional)
    let mut randomized_smiles = None;
    if molecular_representations.contains(&"randomized_smiles".to_string()) {
        buffer.clear();
        reader.read_until(DELIMITER, &mut buffer).ok()?;
        randomized_smiles = Some(String::from_utf8_lossy(&buffer[..buffer.len() - 1]).to_string());
    }

    // Read sns_fp (optional, 16 bytes)
    let mut sns_buf = [0u8; 16];
    if molecular_representations.contains(&"sns".to_string()) {
        reader.read_exact(&mut sns_buf).ok()?;  // Read 16 bytes
        reader.read_exact(&mut delimiter_buf).ok()?;  // Read delimiter
    }

    // Store parsed data
    let smiles_data = SmilesData {
        isomeric_smiles,
        canonical_smiles,
        randomized_smiles,
        target_value,
        sns_buf
    };

    Some(smiles_data)
}

// TODO: Remove logging when you're done testing all molecular representations
fn write_data(
    reader: &mut BufReader<File>,
    writer: &mut BufWriter<File>,
    config: &Config,
    mean: f32,
    std_dev: f32,
    noise_map: &HashMap<usize, f32>,
    tokenizer: &SmilesTokenizer,
    vocab: &HashMap<String, usize>,
    vocab_size: usize,
    max_sequence_length: usize,
    data_count: usize,
    log_writes: bool,  // <-- Added flag for logging
) -> io::Result<()> {

    for index in 0..data_count {
        if let Some(smiles_data) = read_smiles_data(
            reader,
            config.molecular_representations.clone(),
            config.k_domains,
        ) {
            let reader_pos_after = reader.stream_position()?;
            writer.seek(SeekFrom::Start(reader_pos_after))?;

            if log_writes {
                println!("Writing data for index: {}", index);
            }

            // Write isomeric_smiles
            writer.write_all(smiles_data.isomeric_smiles.as_bytes())?;
            writer.write_all(&[DELIMITER])?;
            if log_writes {
                println!("isomeric_smiles: {}", smiles_data.isomeric_smiles);
            }

            // Write canonical_smiles
            writer.write_all(smiles_data.canonical_smiles.as_bytes())?;
            writer.write_all(&[DELIMITER])?;
            if log_writes {
                println!("canonical_smiles: {}", smiles_data.canonical_smiles);
            }

            // Write property_value (float)
            writer.write_all(&smiles_data.target_value.to_le_bytes())?;
            writer.write_all(&[DELIMITER])?;
            if log_writes {
                println!("property_value: {}", smiles_data.target_value);
            }

            // Write randomized_smiles if it exists
            if let Some(randomized) = &smiles_data.randomized_smiles {
                writer.write_all(randomized.as_bytes())?;
                writer.write_all(&[DELIMITER])?;
                if log_writes {
                    println!("randomized_smiles: {}", randomized);
                }
            }

            // Write sns_fp (16 bytes) if applicable
            if config.molecular_representations.contains(&"sns".to_string()) {
                let sns_fp = smiles_data.sns_buf; // sns_buf is already [u8; 16]
                writer.write_all(&sns_fp)?;
                writer.write_all(&[DELIMITER])?;
                if log_writes {
                    println!("sns_fp: {:?}", sns_fp);
                }
            }

            // Normalize and write property value
            // TODO: add noise differently for classification
            let mut property_value = smiles_data.target_value;
            if config.noise {
                if let Some(&artificial_noise) = noise_map.get(&index) {
                    property_value += artificial_noise;
                }
            }

            if config.regression {
                let property_value = (property_value - mean) / std_dev;
            }
            writer.write_all(&property_value.to_le_bytes())?;
            writer.write_all(&[DELIMITER])?;
            if log_writes {
                println!("noisy y: {}", property_value);
            }

            // If multiple domains exist, write domain flag
            if config.k_domains > 1 {
                writer.write_all(&[0u8])?;
                writer.write_all(&[DELIMITER])?;
                if log_writes {
                    println!("domain_flag: 0");
                }
            }


            // Write SMILES and randomized SMILES
            for smiles_type in ["smiles", "randomized_smiles"] {
                if config.molecular_representations.contains(&smiles_type.to_string()) {
                    let smiles_string = if smiles_type == "smiles" {
                        &smiles_data.canonical_smiles
                    } else {
                        smiles_data.randomized_smiles.as_ref().unwrap()
                    };

                    let smiles_ohe = smiles_to_ohe(
                        smiles_string,
                        tokenizer,
                        vocab,
                        vocab_size,
                        max_sequence_length,
                    );
                    let bit_packed_len = (smiles_ohe.len() + 7) / 8;
                    let mut bit_packed_data = vec![0u8; bit_packed_len];

                    for (i, &bit) in smiles_ohe.iter().enumerate() {
                        let byte_index = i / 8;
                        let bit_offset = i % 8;
                        if bit > 0.0 {
                            bit_packed_data[byte_index] |= 1 << bit_offset;
                        }
                    }

                    writer.write_all(&bit_packed_data)?;
                    writer.write_all(&[DELIMITER])?;
                    if log_writes {
                        println!("{}: {:?}", smiles_type, bit_packed_data);
                    }
                }
            }

            // Write ECFP4 fingerprint if applicable
            if config.molecular_representations.contains(&"ecfp4".to_string()) {
                let_cxx_string!(smiles_cxx = smiles_data.isomeric_smiles.clone());
                match smiles_to_mol(&smiles_cxx) {
                    Ok(mol) => {
                        let fingerprint = rdk_fingerprint_mol(&mol);
                        let cxx_vec_ptr: UniquePtr<CxxVector<u64>> = explicit_bit_vect_to_u64_vec(&fingerprint);
                        let cxx_vec_ref: &CxxVector<u64> = &*cxx_vec_ptr;
                        let u64_vec: Vec<u64> = cxx_vec_ref.iter().copied().collect();

                        let mut packed_fingerprint = [0u8; 32];
                        for (i, chunk) in u64_vec.iter().take(4).enumerate() {
                            let byte_index = i * 8;
                            packed_fingerprint[byte_index..byte_index + 8].copy_from_slice(&chunk.to_le_bytes());
                        }

                        writer.write_all(&packed_fingerprint)?;
                        writer.write_all(&[DELIMITER])?;
                        if log_writes {
                            println!("ecfp4_fingerprint: {:?}", packed_fingerprint);
                        }
                    }
                    Err(_) => continue,
                }
            }

            // Ensure writer is properly aligned for next entry
            writer.write_all(b"\n")?;
            writer.flush()?;

            if log_writes {
                println!("Finished writing entry {}\n", index);
            }
        }
    }
    Ok(())
}

fn tanimoto_distance(fp1: &Vec<u64>, fp2: &Vec<u64>) -> f32 {
    let intersection = fp1.iter().zip(fp2.iter()).map(|(&a, &b)| (a & b).count_ones()).sum::<u32>();
    let union = fp1.iter().zip(fp2.iter()).map(|(&a, &b)| (a | b).count_ones()).sum::<u32>();
    1.0 - (intersection as f32 / union as f32)
}

fn smiles_to_ohe(smiles: &str, tokenizer: &SmilesTokenizer, vocab: &HashMap<String, usize>, vocab_size: usize, max_length: usize) -> Array2<f32> {
    let tokens = tokenizer.tokenize(smiles);
    let mut ohe = Array2::<f32>::zeros((max_length, vocab_size));
    for (i, token) in tokens.iter().enumerate().take(max_length) {
        if let Some(&index) = vocab.get(token) {
            if i < max_length {
                ohe[(i, index)] = 1.0;
            }
        }
    }
    ohe
}

fn mean_absolute_error<T: Float + FromPrimitive + Sum<T>>(y_true: &[T], y_pred: &[T]) -> T {
    y_true.iter().zip(y_pred.iter())
          .map(|(true_val, pred_val)| (*true_val - *pred_val).abs())
          .sum::<T>() / T::from_usize(y_true.len()).unwrap()
}

fn mean_squared_error<T: Float + FromPrimitive + Sum<T>>(y_true: &[T], y_pred: &[T]) -> T {
    y_true.iter().zip(y_pred.iter())
          .map(|(true_val, pred_val)| (*true_val - *pred_val).powi(2))
          .sum::<T>() / T::from_usize(y_true.len()).unwrap()
}

fn root_mean_squared_error<T: Float + FromPrimitive + Sum<T>>(y_true: &[T], y_pred: &[T]) -> T {
    mean_squared_error(y_true, y_pred).sqrt()
}

fn r2_score<T: Float + FromPrimitive + Sum<T>>(y_true: &[T], y_pred: &[T]) -> T {
    let mean_true = y_true.iter().map(|&x| x).sum::<T>() / T::from_usize(y_true.len()).unwrap();
    let ss_tot = y_true.iter().map(|&x| (x - mean_true).powi(2)).sum::<T>();
    let ss_res = y_true.iter().zip(y_pred.iter())
                       .map(|(&true_val, &pred_val)| (true_val - pred_val).powi(2))
                       .sum::<T>();
    T::one() - (ss_res / ss_tot)
}

fn count_token_frequencies(smiles_list: &[String], tokenizer: &SmilesTokenizer) -> HashMap<String, usize> {
    let mut token_counts: HashMap<String, usize> = HashMap::new();
    for smiles in smiles_list {
        let tokens = tokenizer.tokenize(smiles);
        for token in tokens {
            *token_counts.entry(token).or_insert(0) += 1;
        }
    }
    token_counts
}

fn trim_vocab<T: Eq + Hash + Ord + Clone>(token_counts: HashMap<T, usize>, max_vocab_size: usize) -> HashMap<T, usize> {
    let mut token_counts_vec: Vec<(T, usize)> = token_counts.into_iter().collect();

    // Sort tokens by count, descending
    token_counts_vec.sort_by_key(|&(_, count)| Reverse(count));

    // Truncate to keep only the top max_vocab_size tokens
    token_counts_vec.truncate(max_vocab_size);

    let trimmed_vocab: HashMap<T, usize> = token_counts_vec.into_iter()
        .enumerate()
        .map(|(idx, (token, _))| (token, idx))
        .collect();

    trimmed_vocab
}

fn generate_aggregate_stats(
    config: &Config,
    noise_map: &HashMap<usize, f32>,
) -> io::Result<(f32, f32, usize, HashMap<String, usize>, usize)> {
    let tokenizer = SmilesTokenizer::new();
    let mut smiles_list: Vec<String> = Vec::new();
    let mut y_values: Vec<f32> = Vec::new();
    let mut max_sequence_length = 0usize;

    let files_to_process = vec![File::open(format!("train_{}.mmap", config.iteration_seed))?];

    for file in files_to_process {
        let mut reader = BufReader::new(file);
        reader.seek(SeekFrom::Start(0))?;

        for index in 0..config.train_count {
            if let Some(smiles_data) = read_smiles_data(&mut reader, config.molecular_representations.clone(), config.k_domains) {
                if config.molecular_representations.contains(&"smiles".to_string()) {
                    smiles_list.push(smiles_data.canonical_smiles.clone());
                    let tokens = tokenizer.tokenize(&smiles_data.canonical_smiles);
                    max_sequence_length = std::cmp::max(max_sequence_length, tokens.len());
                }

                let mut property_value = smiles_data.target_value;
                if config.noise {
                    // TODO: add logic back in when you have domain labels again
                    // if matches!(config.noise_distribution, NoiseDistribution::DomainMpnn | NoiseDistribution::DomainTanimoto) {
                    //     if smiles_data.domain_label == config.target_domain as i32 {
                    //         // Apply noise only if the domain matches the target domain
                    //         if let Some(&artificial_noise) = noise_map.get(&index) {
                    //             property_value += artificial_noise;
                    //         }
                    //     }
                    // } else {
                    //     // Apply noise from the noise map for other distributions
                    //     if let Some(&artificial_noise) = noise_map.get(&index) {
                    //         property_value += artificial_noise;
                    //     }
                    // }
                    // And then delete this:
                    if let Some(&artificial_noise) = noise_map.get(&index) {
                        property_value += artificial_noise;
                    }
                }
                y_values.push(property_value);
            }
        }
    }

    let token_counts = count_token_frequencies(&smiles_list, &tokenizer);

    let trimmed_vocab = trim_vocab(token_counts, config.max_vocab);
    let vocab_size = trimmed_vocab.len();

    let mean: f32 = y_values.iter().sum::<f32>() / y_values.len() as f32;
    let variance: f32 = y_values.iter().map(|value| {
        let diff = mean - *value;
        diff * diff
    }).sum::<f32>() / y_values.len() as f32;
    let std_deviation: f32 = variance.sqrt();

    Ok((mean, variance, vocab_size, trimmed_vocab, max_sequence_length))
}

fn preprocess_data(
    config: &Config,
    mean: f32,
    std_dev: f32,
    vocab_size: usize,
    vocab: &HashMap<String, usize>,
    noise_map: &HashMap<usize, f32>,
    max_sequence_length: usize,
) -> io::Result<()> {
    let tokenizer = SmilesTokenizer::new();

    let train_file_path = format!("train_{}.mmap", config.iteration_seed);
    let test_file_path = format!("test_{}.mmap", config.iteration_seed);

    let train_file = File::open(&train_file_path)?;
    let test_file = File::open(&test_file_path)?;

    let mut train_reader = BufReader::new(train_file);
    let mut train_writer = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .append(true)
            .open(&train_file_path)
            .unwrap()
    );
    train_reader.seek(SeekFrom::Start(0))?;
    write_data(
        &mut train_reader,
        &mut train_writer,
        config,
        mean,
        std_dev,
        noise_map,
        &tokenizer,
        vocab,
        vocab_size,
        max_sequence_length,
        config.train_count,
        config.logging,
    )?;

    let mut test_reader = BufReader::new(test_file);
    let mut test_writer = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .append(true)
            .open(&test_file_path)
            .unwrap()
    );
    test_reader.seek(SeekFrom::Start(0))?;
    write_data(
        &mut test_reader,
        &mut test_writer,
        config,
        mean,
        std_dev,
        noise_map,
        &tokenizer,
        vocab,
        vocab_size,
        max_sequence_length,
        config.test_count,
        config.logging,
    )?;

    Ok(())
}

// TODO: re-introduce params for other distributions
fn main() -> io::Result<()> {
    // env::set_var("DYLD_LIBRARY_PATH", "/usr/local/Cellar/libtensorflow/2.15.0");

    let app = Command::new("My Rust Processor")
        .arg(Arg::new("seed")
             .long("seed")
             .action(ArgAction::Set)
             .help("Random seed for the process"))
        .arg(Arg::new("model")
             .long("model")
             .action(ArgAction::Set)
             .help("Model to use for prediction"))
        .arg(Arg::new("sigma")
             .long("sigma")
             .action(ArgAction::Set)
             .help("Sigma for artificial noise addition"))
        // .arg(Arg::new("sampling_proportion")
        //      .long("sampling_proportion")
        //      .action(ArgAction::Set)
        //      .help("Sampling proportion for artificial noise addition"))
        // .arg(Arg::new("noise_mu")
        //      .long("noise_mu")
        //      .action(ArgAction::Set)
        //      .help("Mean for noise distribution"))
        .arg(Arg::new("noise_distribution")
             .long("noise_distribution")
             .action(ArgAction::Set)
             .help("Distribution type for noise"));
        // .arg(Arg::new("alpha_skew")
        //      .long("alpha_skew")
        //      .action(ArgAction::Set)
        //      .help("Alpha skew for noise distribution"))
        // .arg(Arg::new("beta_param1")
        //      .long("beta_param1")
        //      .action(ArgAction::Set)
        //      .help("First parameter for beta distribution"))
        // .arg(Arg::new("beta_param2")
        //      .long("beta_param2")
        //      .action(ArgAction::Set)
        //      .help("Second parameter for beta distribution"));

    let matches = app.get_matches();

    let seed: u64 = matches.get_one::<String>("seed")
                           .unwrap()
                           .parse()
                           .expect("Seed must be a valid integer");
    let model = matches.get_one::<String>("model").unwrap();
    let sigma: f32 = matches.get_one::<String>("sigma").unwrap().parse().expect("Sigma must be a valid float");
    // let sampling_proportion: f32 = matches.get_one::<String>("sampling_proportion").unwrap().parse().expect("Sampling proportion must be a valid float");
    // let noise_mu: f32 = matches.get_one::<String>("noise_mu").unwrap().parse().expect("Noise mu must be a valid float");
    let noise_distribution: NoiseDistribution = match matches.get_one::<String>("noise_distribution").unwrap().as_str() {
        "gaussian" => NoiseDistribution::Gaussian,
        "left-tailed" => NoiseDistribution::LeftTailed,
        "right-tailed" => NoiseDistribution::RightTailed,
        "u-shaped" => NoiseDistribution::UShaped,
        "uniform" => NoiseDistribution::Uniform,
        "domain_mpnn" => NoiseDistribution::DomainMpnn,
        "domain_tanimoto" => NoiseDistribution::DomainTanimoto,
        _ => panic!("Invalid noise distribution specified"),
    };
    // let alpha_skew: f32 = matches.get_one::<String>("alpha_skew").unwrap().parse().expect("Alpha skew must be a valid float");
    // let beta_param1: f32 = matches.get_one::<String>("beta_param1").unwrap().parse().expect("Beta param1 must be a valid float");
    // let beta_param2: f32 = matches.get_one::<String>("beta_param2").unwrap().parse().expect("Beta param2 must be a valid float");
    let alpha_skew: f32 = 0.0;
    let beta_param1: f32 = 0.5;
    let beta_param2: f32 = 0.5;
    let noise_mu: f32 = 0.0;
    // TODO: potentially change this
    let sampling_proportion: f32 = 1.0;

    // Reading the configuration file
    let config_file = File::open("config.json")?;
    let reader = BufReader::new(config_file);
    let config: Config = serde_json::from_reader(reader)
                          .expect("JSON was not well-formatted or did not match the expected structure");

    let noise_indices: Vec<usize> = if config.noise {
        (0..config.train_count)
            .filter(|_| rand::random::<f32>() < sampling_proportion)
            .collect()
    } else {
        Vec::new()
    };

    let noise_map: HashMap<usize, f32> = generate_noise_by_indices(
        &noise_indices,
        noise_mu,
        sigma,
        noise_distribution.clone(),
        alpha_skew,
        (beta_param1, beta_param2),
        seed,
    );

    let (mean, std_dev, vocab_size, vocab, max_sequence_length) =
        generate_aggregate_stats(&config, &noise_map)?;

    // println!("Starting data preprocessing...");
    preprocess_data(
        &config,
        mean,
        std_dev,
        vocab_size,
        &vocab,
        &noise_map,
        max_sequence_length
    )?;
    // println!("Data preprocessing complete");

    Ok(())
}

// TODO: americanize code