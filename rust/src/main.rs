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
use std::fs::{OpenOptions, remove_file, rename};
use rand::SeedableRng;

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

#[derive(Deserialize, Debug)]
struct Config {
    sample_size: usize,
    noise: bool,
    train_count: usize, 
    test_count: usize,
    val_count: usize,
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
    sns_buf: [u8; 128]
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
    target_sigma: f32,
    distribution: NoiseDistribution,
    seed: u64,
) -> HashMap<usize, f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut noise_map = HashMap::new();

    for &idx in indices {
        let noise = match distribution {
            NoiseDistribution::Gaussian | NoiseDistribution::DomainMpnn | NoiseDistribution::DomainTanimoto => {
                let normal = Normal::new(0.0, target_sigma).unwrap();
                normal.sample(&mut rng)
            }
            NoiseDistribution::LeftTailed => {
                let alpha = 5.0; // fixed skew strength
                let skew_normal = SkewNormal::new(-alpha, 0.0, target_sigma).unwrap();
                skew_normal.sample(&mut rng)
            }
            NoiseDistribution::RightTailed => {
                let alpha = 5.0; // fixed skew strength
                let skew_normal = SkewNormal::new(alpha, 0.0, target_sigma).unwrap();
                skew_normal.sample(&mut rng)
            }
            NoiseDistribution::UShaped => {
                // Beta(0.5, 0.5) is U-shaped; rescale to [-1,1]
                let beta = Beta::new(1.5, 1.5).unwrap();
                let sample = beta.sample(&mut rng);
                let scaling_factor = target_sigma / 0.70710678118;
                (sample - 0.5) * 2.0 * scaling_factor
            }
            NoiseDistribution::Uniform => {
                // Uniform between [-a, a], variance = a² / 3
                // Solve a²/3 = target_sigma²
                let a = (3.0 * target_sigma.powi(2)).sqrt();
                let uniform = Uniform::new_inclusive(-a, a).unwrap();
                uniform.sample(&mut rng)
            }
        };

        noise_map.insert(idx, noise as f32);
    }

    noise_map
}

fn read_smiles_data(
    reader: &mut BufReader<File>,
    molecular_representations: Vec<String>,
    _k_domains: usize,
) -> Option<SmilesData> {
    // Helper: read a 4-byte length-prefixed UTF-8 string
    fn read_len_prefixed_string(reader: &mut BufReader<File>) -> Option<String> {
        let mut len_buf = [0u8; 4];
        reader.read_exact(&mut len_buf).ok()?;
        let str_len = u32::from_le_bytes(len_buf) as usize;
        let mut buf = vec![0u8; str_len];
        reader.read_exact(&mut buf).ok()?;
        String::from_utf8(buf).ok()
    }

    // Read isomeric_smiles and check validity
    let isomeric_smiles = read_len_prefixed_string(reader)?;
    if isomeric_smiles.len() < 5 || isomeric_smiles.len() > 300 || isomeric_smiles.contains(['\u{FFFD}', '\0', '\'', '�']) {
        eprintln!("Skipping malformed isomeric_smiles: {:?}", isomeric_smiles);
        return None;
    }

    // Read canonical_smiles
    let canonical_smiles = read_len_prefixed_string(reader)?;

    // Read property_value (float)
    let mut prop_buf = [0u8; 4];
    reader.read_exact(&mut prop_buf).ok()?;
    let target_value = f32::from_le_bytes(prop_buf);

    // Read randomized_smiles if applicable
    let mut randomized_smiles = None;
    if molecular_representations.contains(&"randomized_smiles".to_string()) {
        let mut len_buf = [0u8; 4];
        reader.read_exact(&mut len_buf).ok()?;
        let rand_len = u32::from_le_bytes(len_buf) as usize;
        if rand_len > 0 {
            let mut rand_buf = vec![0u8; rand_len];
            reader.read_exact(&mut rand_buf).ok()?;
            randomized_smiles = String::from_utf8(rand_buf).ok();
        }
    }

    // Read sns_fp if applicable
    let mut sns_buf = [0u8; 128];
    if molecular_representations.contains(&"sns".to_string()) {
        reader.read_exact(&mut sns_buf).ok()?;
    }

    Some(SmilesData {
        isomeric_smiles,
        canonical_smiles,
        randomized_smiles,
        target_value,
        sns_buf,
    })
}

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
    log_writes: bool,
) -> io::Result<()> {
    for index in 0..data_count {
        if let Some(smiles_data) = read_smiles_data(
            reader,
            config.molecular_representations.clone(),
            config.k_domains,
        ) {

            if log_writes {
                println!("Writing data for index: {}", index);
            }

            // Write isomeric_smiles with length prefix
            let iso_bytes = smiles_data.isomeric_smiles.as_bytes();
            let iso_len_bytes = (iso_bytes.len() as u32).to_le_bytes();
            writer.write_all(&iso_len_bytes)?;
            writer.write_all(iso_bytes)?;
            if log_writes {
                println!("isomeric_smiles: {}", smiles_data.isomeric_smiles);
                println!("isomeric_smiles_len bytes: {:02X?}", iso_len_bytes);
                println!("isomeric_smiles bytes: {:02X?}", iso_bytes);
            }

            // Write canonical_smiles with length prefix
            let canon_bytes = smiles_data.canonical_smiles.as_bytes();
            let canon_len_bytes = (canon_bytes.len() as u32).to_le_bytes();
            writer.write_all(&canon_len_bytes)?;
            writer.write_all(canon_bytes)?;
            if log_writes {
                println!("canonical_smiles: {}", smiles_data.canonical_smiles);
                println!("canonical_smiles_len bytes: {:02X?}", canon_len_bytes);
                println!("canonical_smiles bytes: {:02X?}", canon_bytes);
            }

            // Write target value
            let target_bytes = smiles_data.target_value.to_le_bytes();
            writer.write_all(&target_bytes)?;
            if log_writes {
                println!("property_value: {}", smiles_data.target_value);
                println!("property_value bytes: {:02X?}", target_bytes);
            }

            // Write randomized_smiles if exists
            if let Some(randomized) = &smiles_data.randomized_smiles {
                let bytes = randomized.as_bytes();
                let len_bytes = (bytes.len() as u32).to_le_bytes();
                writer.write_all(&len_bytes)?;
                writer.write_all(bytes)?;
                if log_writes {
                    println!("randomized_smiles: {}", randomized);
                    println!("randomized_smiles_len bytes: {:02X?}", len_bytes);
                    println!("randomized_smiles bytes: {:02X?}", bytes);
                }
            }

            // Write sns_fp
            if config.molecular_representations.contains(&"sns".to_string()) {
                let sns_fp = smiles_data.sns_buf;
                writer.write_all(&sns_fp)?;
                if log_writes {
                    println!("sns_fp: {:?}", sns_fp);
                    println!("sns_fp bytes: {:02X?}", sns_fp);
                }
            }

            // Normalize and write processed target
            let mut property_value = smiles_data.target_value;
            if config.noise {
                if let Some(&artificial_noise) = noise_map.get(&index) {
                    property_value += artificial_noise;
                }
            }

            if config.regression {
                property_value = (property_value - mean) / std_dev;
            }

            let processed_bytes = property_value.to_le_bytes();
            writer.write_all(&processed_bytes)?;
            if log_writes {
                println!("noisy y: {}", property_value);
                println!("noisy y bytes: {:02X?}", processed_bytes);
            }

            // Write domain label if applicable
            if config.k_domains > 1 {
                writer.write_all(&[0u8])?;
                if log_writes {
                    println!("domain_flag: 0");
                    println!("domain_flag bytes: 00");
                }
            }

            // Write smiles or randomized_smiles OHE if used
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

                    let len_bytes = (bit_packed_data.len() as u32).to_le_bytes();
                    writer.write_all(&len_bytes)?;
                    writer.write_all(&bit_packed_data)?;
                    if log_writes {
                        println!("{}: {:?}", smiles_type, bit_packed_data);
                        println!("{}_ohe_len bytes: {:02X?}", smiles_type, len_bytes);
                        println!("{}_ohe bytes: {:02X?}", smiles_type, bit_packed_data);
                    }
                }
            }

            // Write ECFP4 fingerprint
            if config.molecular_representations.contains(&"ecfp4".to_string()) {
                let_cxx_string!(smiles_cxx = smiles_data.isomeric_smiles.clone());
                match smiles_to_mol(&smiles_cxx) {
                    Ok(mol) => {
                        let fingerprint = rdk_fingerprint_mol(&mol);
                        let cxx_vec_ptr: UniquePtr<CxxVector<u64>> = explicit_bit_vect_to_u64_vec(&fingerprint);
                        let cxx_vec_ref: &CxxVector<u64> = &*cxx_vec_ptr;
                        let mut u64_vec: Vec<u64> = cxx_vec_ref.iter().copied().collect();

                        if u64_vec.len() != 32 {
                            if log_writes {
                                eprintln!("Index {}: ECFP4 is not 2048 bits! Got {} chunks", index, u64_vec.len());
                            }
                            continue;
                        }

                        let mut packed_fingerprint = vec![0u8; 256];
                        for (i, chunk) in u64_vec.iter().enumerate() {
                            packed_fingerprint[i * 8..(i + 1) * 8].copy_from_slice(&chunk.to_le_bytes());
                        }

                        writer.write_all(&packed_fingerprint)?;
                        if log_writes {
                            println!("ecfp4_fingerprint: {:?}", packed_fingerprint);
                        }
                    }
                    Err(_) => {
                        if log_writes {
                            eprintln!("Skipping index {}: Failed to parse mol", index);
                        }
                        continue;
                    }
                }
            }

            if log_writes {
                println!("Finished writing entry {}\n", index);
            }
        }
    }

    writer.flush()?;

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
                if ["smiles", "randomized_smiles"].iter().any(|r| config.molecular_representations.contains(&r.to_string())) {
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
    let train_file_new_path = format!("train_{}_new.mmap", config.iteration_seed);
    let test_file_path = format!("test_{}.mmap", config.iteration_seed);
    let test_file_new_path = format!("test_{}_new.mmap", config.iteration_seed);
    let val_file_path = format!("val_{}.mmap", config.iteration_seed);
    let val_file_new_path = format!("val_{}_new.mmap", config.iteration_seed);

    let train_file = File::open(&train_file_path)?;
    let test_file = File::open(&test_file_path)?;
    let val_file = File::open(&val_file_path)?;

    let mut train_reader = BufReader::new(train_file);
    let mut train_writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&train_file_new_path)?
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
    remove_file(&train_file_path)?;
    rename(&train_file_new_path, &train_file_path)?;

    let mut val_reader = BufReader::new(val_file);
    let mut val_writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&val_file_new_path)?
    );
    val_reader.seek(SeekFrom::Start(0))?;
    write_data(
        &mut val_reader,
        &mut val_writer,
        config,
        mean,
        std_dev,
        noise_map,
        &tokenizer,
        vocab,
        vocab_size,
        max_sequence_length,
        config.val_count,
        config.logging,
    )?;
    remove_file(&val_file_path)?;
    rename(&val_file_new_path, &val_file_path)?;

    let mut test_reader = BufReader::new(test_file);
    let mut test_writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&test_file_new_path)?
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
    remove_file(&test_file_path)?;
    rename(&test_file_new_path, &test_file_path)?;

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
        .arg(Arg::new("noise_distribution")
             .long("noise_distribution")
             .action(ArgAction::Set)
             .help("Distribution type for noise"));

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
        sigma,
        noise_distribution.clone(),
        seed,
    );

    let (mean, std_dev, vocab_size, vocab, max_sequence_length) =
        generate_aggregate_stats(&config, &noise_map)?;

    preprocess_data(
        &config,
        mean,
        std_dev,
        vocab_size,
        &vocab,
        &noise_map,
        max_sequence_length
    )?;

    Ok(())
}

// TODO: americanize code