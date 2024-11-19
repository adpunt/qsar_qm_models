use std::collections::HashMap;
use std::hash::Hash;
use regex::Regex;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use serde_json::json;
use ndarray::Array2;
use std::fs::File;
use serde::{Deserialize, Serialize};
use clap::{Arg, Command, ArgAction};
use num_traits::{Float, FromPrimitive};
use std::iter::Sum;
use rand_distr::{Distribution, Normal, Uniform, Beta, SkewNormal};
use rand::SeedableRng;
use std::thread;
use std::env;
use rand::Rng;
use std::io::Write;
use std::cmp::Reverse;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyAny, PyTuple};

extern crate rdkit_sys;
// extern crate tensorflow;

// use tensorflow::Tensor;

use rdkit_sys::ro_mol_ffi::{smiles_to_mol};
use rdkit_sys::fingerprint_ffi::{fingerprint_mol, explicit_bit_vect_to_u64_vec}; // Assuming fingerprint generation is related to this type.
use cxx::let_cxx_string;
use cxx::UniquePtr;
use cxx::CxxVector;

// Todo: add this back in later: xgboost = "0.1.4"
// use xgboost::{
//     parameters::TrainingParametersBuilder,
//     DMatrix, 
//     Booster, 
//     XGBError
// };

use vega_lite_4::{VegaliteBuilder, Mark, XClassBuilder, YClassBuilder, EdEncodingBuilder, Type, Showable};

use std::result::Result as XGBResult;

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

#[derive(Serialize, Clone, Debug)]
enum Fingerprint {
    ECFP4(Vec<u64>),
    SMILES(Array2<f32>),
    SRAW(String),
}

impl Fingerprint {
    pub fn to_feature_array(&self) -> Vec<f32> {
        match self {
            Fingerprint::ECFP4(bits) => {
                // Convert Vec<u64> to Vec<f32>
                // Flattening u64 bits to a vector of f32 where each bit is a feature
                bits.iter()
                    .flat_map(|&num| {
                        (0..64).map(move |i| {
                            if num & (1 << i) != 0 { 1.0 } else { 0.0 }
                        })
                    })
                    .collect()
            },
            Fingerprint::SMILES(matrix) => {
                // Flatten Array2<f32> to Vec<f32>
                matrix.iter().cloned().collect()
            },
            Fingerprint::SRAW(smiles) => {
                // Convert SMILES string to feature array
                smiles.chars()
                    .map(|c| (c as u8 as f32) / 255.0)  // Normalize ASCII value to [0, 1]
                    .collect()
            },
        }
    }
}

#[derive(Deserialize, Debug)]
struct Config {
    sample_size: usize,
    noise: bool,
    train_count: usize, 
    test_count: usize,
    max_vocab: usize,
    bootstrapping_iteration: usize,
    target_domain: usize,
}

struct SmilesData {
    isomeric_smiles: String,
    canonical_smiles: String,
    alternative_smiles: String,
    qm_property_value: f64,
    domain_label: i32,
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

fn ecfp_atom_ids_from_smiles(smiles: &str) -> PyResult<HashMap<u64, usize>> {
    Python::with_gil(|py| {
        // Import the sys module
        let sys = PyModule::import(py, "sys")?;

        // Get the sys.path attribute
        let sys_path: &PyAny = sys.getattr("path")?;

        // Convert the current directory to a Python string and append it to sys.path
        let current_dir = env::current_dir().unwrap();
        let current_dir_str = current_dir.to_str().unwrap().to_object(py);
        sys_path.call_method1("append", (current_dir_str,))?;

        // Attempt to import the sns_short Python module
        let sns_module = PyModule::import(py, "sns_short")
            .map_err(|e| {
                e.print(py);
                PyErr::new::<pyo3::exceptions::PyImportError, _>("Failed to import sns_short module")
            })?;

        // Convert the SMILES string to a Python object
        let smiles_py = smiles.to_object(py);

        // Create a tuple to pass as arguments
        let args = PyTuple::new(py, &[smiles_py]);

        // Call the Python function with the arguments
        let result = sns_module.getattr("ecfp_atom_ids_from_smiles")?.call1(args)?;

        // Extract the result to a Rust HashMap
        let substructure_dict: HashMap<u64, usize> = result.extract()?;
        Ok(substructure_dict)
    })
}

fn generate_noise_by_indices(
    indices: &[usize],
    mu: f64,
    sigma: f64,
    distribution: NoiseDistribution,
    alpha_skew: f64,
    beta_params: (f64, f64),
    seed: u64,
) -> HashMap<usize, f64> {
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

        noise_map.insert(idx, noise);
    }

    noise_map
}

fn send_to_python(fps_train: Vec<Fingerprint>, y_train: Vec<f64>, fps_test: Vec<Fingerprint>, y_fixed_test: Vec<f64>) {
    // Generate a unique identifier for the file
    let thread_id = thread::current().id();
    let random_number: u64 = rand::thread_rng().gen();
    let unique_id = format!("{}", random_number);

    // Serialize the data to JSON
    let data = json!({
        "fps_train": fps_train,
        "y_train": y_train,
        "fps_test": fps_test,
        "y_fixed_test": y_fixed_test
    });

    // Convert the JSON data to a string
    let json_string = data.to_string();

    // Create a temporary file path with the unique identifier
    let temp_dir = env::temp_dir();
    let temp_file_path = temp_dir.join(format!("rust_to_python_data_{}.json", unique_id));

    // Write the JSON string to the temporary file
    let mut file = File::create(&temp_file_path).expect("Unable to create temporary file");
    file.write_all(json_string.as_bytes()).expect("Unable to write data to file");

    // Print the file path to stdout for the Python script to read, with a specific prefix
    println!("TEMP_FILE_PATH:{}", temp_file_path.to_string_lossy());
}

fn read_smiles_data(reader: &mut BufReader<File>) -> Option<SmilesData> {
    let mut length_buf = [0u8; 4];

    // Read the length of the isomeric SMILES string
    if reader.read_exact(&mut length_buf).is_err() {
        return None;
    }
    let isomeric_length = u32::from_le_bytes(length_buf) as usize;
    let mut isomeric_smiles_buf = vec![0u8; isomeric_length];
    if reader.read_exact(&mut isomeric_smiles_buf).is_err() {
        return None;
    }

    // Read the length of the canonical SMILES string
    if reader.read_exact(&mut length_buf).is_err() {
        return None;
    }
    let canonical_length = u32::from_le_bytes(length_buf) as usize;
    let mut canonical_smiles_buf = vec![0u8; canonical_length];
    if reader.read_exact(&mut canonical_smiles_buf).is_err() {
        return None;
    }

    // Read the length of the alternative SMILES string
    if reader.read_exact(&mut length_buf).is_err() {
        return None;
    }
    let alternative_length = u32::from_le_bytes(length_buf) as usize;
    let mut alternative_smiles_buf = vec![0u8; alternative_length];
    if reader.read_exact(&mut alternative_smiles_buf).is_err() {
        return None;
    }

    // Read the property value
    let mut property_buf = [0u8; 8];
    if reader.read_exact(&mut property_buf).is_err() {
        return None;
    }
    let property = f64::from_le_bytes(property_buf);

    // Read the domain label
    if reader.read_exact(&mut length_buf).is_err() {
        return None;
    }
    let domain_label = i32::from_le_bytes(length_buf);

    // Convert SMILES data from bytes to UTF-8 strings
    let isomeric_smiles = match std::str::from_utf8(&isomeric_smiles_buf) {
        Ok(smiles) => smiles.to_string(),
        Err(_) => return None,
    };

    let canonical_smiles = match std::str::from_utf8(&canonical_smiles_buf) {
        Ok(smiles) => smiles.to_string(),
        Err(_) => return None,
    };

    let alternative_smiles = match std::str::from_utf8(&alternative_smiles_buf) {
        Ok(smiles) => smiles.to_string(),
        Err(_) => return None,
    };

    // println!("isomeric_smiles: {:?}", isomeric_smiles);
    // println!("canonical_smiles: {:?}", canonical_smiles);
    // println!("alternative_smiles: {:?}", alternative_smiles);
    // println!("qm_property_value: {:?}", property);
    // println!("domain_label: {:?}", domain_label);

    Some(SmilesData {
        isomeric_smiles,
        canonical_smiles,
        alternative_smiles,
        qm_property_value: property,
        domain_label,
    })
}

// Function to print a few sample fingerprints for verification
fn print_sample_fingerprints(fingerprints: &[Fingerprint], label: &str) {
    println!("Sample fingerprints for {}:", label);
    for (index, fp) in fingerprints.iter().enumerate().take(5) { // Print only the first 5
        println!("  Fingerprint {}: {:?}", index + 1, fp.to_feature_array());
    }
}

fn tanimoto_distance(fp1: &Vec<u64>, fp2: &Vec<u64>) -> f64 {
    let intersection = fp1.iter().zip(fp2.iter()).map(|(&a, &b)| (a & b).count_ones()).sum::<u32>();
    let union = fp1.iter().zip(fp2.iter()).map(|(&a, &b)| (a | b).count_ones()).sum::<u32>();
    1.0 - (intersection as f64 / union as f64)
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

// fn smiles_to_dmatrix(fps: Vec<Fingerprint>, y_train: Vec<f64>, vocab_size: usize) -> DMatrix {
//     let num_features = vocab_size;
//     let mut features = Vec::new();

//     for fp in &fps {
//         if let Fingerprint::SMILES(matrix) = fp {
//             let fp_features: Vec<f32> = matrix.iter().cloned().collect();
//             // Assumes that fp_features always has the correct length
//             features.extend(fp_features);
//         } else {
//             panic!("Unsupported fingerprint type");
//         }
//     }

//     // Convert f64 labels to f32
//     let y_train_f32: Vec<f32> = y_train.iter().map(|&x| x as f32).collect();
    
//     // Create the DMatrix
//     let mut dmatrix = DMatrix::from_dense(&features, fps.len())
//         .expect("Failed to create DMatrix from features"); // Correctly handle the result here

//     dmatrix.set_labels(&y_train_f32)
//         .expect("Failed to set labels for DMatrix");

//     dmatrix // Return the DMatrix
// }

// fn ecfp4_to_dmatrix(fps: Vec<Fingerprint>, y_train: Vec<f64>) -> XGBResult<DMatrix, XGBError> {
//     // Flatten all ECFP4 fingerprints into a single Vec<f32>
//     // Each u64 in the vector is assumed to represent multiple features already suitable for the DMatrix
//     let flattened: Vec<f32> = fps.iter().flat_map(|fp| {
//         match fp {
//             Fingerprint::ECFP4(vec) => {
//                 // Convert each u64 into multiple f32s
//                 vec.iter().flat_map(|&val| {
//                     (0..64).map(move |i| {
//                         if val & (1 << i) != 0 { 1.0 } else { 0.0 }
//                     })
//                 }).collect::<Vec<f32>>()
//             },
//             _ => vec![]
//         }
//     }).collect();

//     let rows = fps.len();  // Number of fingerprints
//     let cols = if !flattened.is_empty() {
//         flattened.len() / rows  // Calculate number of columns based on total features and number of rows
//     } else {
//         0
//     };

//     // Create DMatrix from the flattened feature vector
//     let mut dmatrix = DMatrix::from_dense(&flattened, rows)?;
//     let y_train_slice_f32: Vec<f32> = y_train.iter().map(|&x| x as f32).collect();
//     dmatrix.set_labels(&y_train_slice_f32)?;
//     Ok(dmatrix)
// }

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

fn plot_actual_vs_predicted_vega_f32(actual: &[f32], predicted: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let values: Array2<f32> = Array2::from_shape_fn((actual.len(), 2), |(i, j)| {
        if j == 0 { actual[i] } else { predicted[i] }
    });

    let chart = VegaliteBuilder::default()
        .title("Actual vs. Predicted")
        .data(values)
        .mark(Mark::Point)
        .encoding(
            EdEncodingBuilder::default()
                .x(XClassBuilder::default()
                    .field("data.0")
                    .position_def_type(Type::Quantitative)
                    .build()?)
                .y(YClassBuilder::default()
                    .field("data.1")
                    .position_def_type(Type::Quantitative)
                    .build()?)
                .build()?,
        )
        .build()?;

    chart.show()?;
    Ok(())
}

fn plot_actual_vs_predicted_vega_f64(actual: &[f64], predicted: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    let values: Array2<f64> = Array2::from_shape_fn((actual.len(), 2), |(i, j)| {
        if j == 0 { actual[i] } else { predicted[i] }
    });

    let chart = VegaliteBuilder::default()
        .title("Actual vs. Predicted")
        .data(values)
        .mark(Mark::Point)
        .encoding(
            EdEncodingBuilder::default()
                .x(XClassBuilder::default()
                    .field("data.0")
                    .position_def_type(Type::Quantitative)
                    .build()?)
                .y(YClassBuilder::default()
                    .field("data.1")
                    .position_def_type(Type::Quantitative)
                    .build()?)
                .build()?,
        )
        .build()?;

    chart.show()?;
    Ok(())
}

// fn train_and_predict_with_xgboost(
//     train_dmatrix: DMatrix,
//     test_dmatrix: DMatrix,
//     num_boost_round: u32,
// ) -> XGBResult<(Vec<f32>, f32), XGBError> {

//     // Specify datasets to evaluate against during training
//     let evaluation_sets = &[(&train_dmatrix, "train"), (&test_dmatrix, "test")];

//     // Set up overall training parameters
//     let training_params = TrainingParametersBuilder::default()
//         .dtrain(&train_dmatrix)
//         .boost_rounds(num_boost_round)
//         .evaluation_sets(Some(evaluation_sets))
//         .build()
//         .expect("Failed to build training parameters");

//     // Train the model using the specified parameters
//     let bst = Booster::train(&training_params).expect("Failed to train model");

//     // Predict using the test DMatrix
//     let predictions = bst.predict(&test_dmatrix)?;

//     // Print the predictions
//     // println!("Predictions: {:?}", predictions);

//     let test_labels = test_dmatrix.get_labels().expect("Failed to get test labels");

//     let mae = mean_absolute_error(&test_labels, &predictions);
//     let mse = mean_squared_error(&test_labels, &predictions);
//     let rmse = root_mean_squared_error(&test_labels, &predictions);
//     let r2 = r2_score(&test_labels, &predictions);

//     // Print the computed metrics
//     // println!("MAE: {}", mae);
//     // println!("MSE: {}", mse);
//     // println!("RMSE: {}", rmse);
//     // println!("R^2 Score: {}", r2);

//     if let Err(e) = plot_actual_vs_predicted_vega_f32(&test_labels, &predictions) {
//         eprintln!("Failed to create Vega-lite plot: {:?}", e);
//     }

//     Ok((predictions, r2))
// }


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
    seed: u64,
    molecular_representation: &str,
    train_count: usize,
    sample_size: usize,
    noise: bool,
    noise_map: &HashMap<usize, f64>,
    max_vocab: usize,
    bootstrapping_iteration: usize,
    target_domain: usize,
    distribution: NoiseDistribution,
) -> io::Result<(f64, f64, usize, HashMap<String, usize>, usize, HashMap<u64, usize>, usize, Vec<u64>)> {
    let tokenizer = SmilesTokenizer::new();
    let mut smiles_list: Vec<String> = Vec::new();
    let mut substructure_counts = HashMap::new();
    let mut y_values: Vec<f64> = Vec::new();
    let mut max_sequence_length = 0usize;
    let mut substructure_support = HashMap::new();
    

    let mut files_to_process = vec![File::open(format!("train_{}.mmap", bootstrapping_iteration))?];

    for file in files_to_process {
        let mut reader = BufReader::new(file);
        reader.seek(SeekFrom::Start(0))?;

        for index in 0..train_count {
            if let Some(smiles_data) = read_smiles_data(&mut reader) {
                if molecular_representation == "smiles" {
                    smiles_list.push(smiles_data.canonical_smiles.clone());
                    let tokens = tokenizer.tokenize(&smiles_data.canonical_smiles);
                    max_sequence_length = std::cmp::max(max_sequence_length, tokens.len());
                }

                if molecular_representation == "sns" {
                    let substructures = ecfp_atom_ids_from_smiles(&smiles_data.canonical_smiles).unwrap();  // Using unwrap for simplicity
                    for (sub, count) in substructures {
                        *substructure_counts.entry(sub).or_insert(0) += count;
                        *substructure_support.entry(sub).or_insert(0) += 1;
                    }
                }

                let mut property_value = smiles_data.qm_property_value;
                if noise {
                    if matches!(distribution, NoiseDistribution::DomainMpnn | NoiseDistribution::DomainTanimoto) {
                        if smiles_data.domain_label == target_domain as i32 {
                            // Apply noise only if the domain matches the target domain
                            if let Some(&artificial_noise) = noise_map.get(&index) {
                                property_value += artificial_noise;
                            }
                        }
                    } else {
                        // Apply noise from the noise map for other distributions
                        if let Some(&artificial_noise) = noise_map.get(&index) {
                            property_value += artificial_noise;
                        }
                    }
                }
                y_values.push(property_value);
            }
        }
    }

    let token_counts = count_token_frequencies(&smiles_list, &tokenizer);

    let trimmed_vocab = trim_vocab(token_counts, max_vocab);
    let vocab_size = trimmed_vocab.len();

    let trimmed_substructure_vocab = trim_vocab(substructure_counts, max_vocab);
    let substructure_vocab_size = trimmed_substructure_vocab.len();

    let mut sorted_atom_ids: Vec<_> = substructure_support.iter().collect();
    sorted_atom_ids.sort_by(|a, b| b.1.cmp(a.1)); // Sort by support, descending

    let final_atom_id_list: Vec<u64> = sorted_atom_ids.iter()
        .take(max_vocab)
        .map(|(atom_id, _)| **atom_id)
        .collect();

    let mean: f64 = y_values.iter().sum::<f64>() / y_values.len() as f64;
    let variance: f64 = y_values.iter().map(|value| {
        let diff = mean - *value;
        diff * diff
    }).sum::<f64>() / y_values.len() as f64;
    let std_deviation: f64 = variance.sqrt();

    let vocab_json = json!({
        "mean": mean, 
        "std_dev": std_deviation, 
        "vocab_size": vocab_size, 
        "vocab": trimmed_vocab
    });

    Ok((mean, variance, vocab_size, trimmed_vocab, max_sequence_length, trimmed_substructure_vocab, substructure_vocab_size, final_atom_id_list))
}

fn preprocess_data(
    seed: u64,
    molecular_representation: &str,
    model: &str,
    sample_size: usize,
    mean: f64,
    std_dev: f64,
    vocab_size: usize,
    vocab: &HashMap<String, usize>,
    substructure_vocab: &HashMap<u64, usize>,
    substructure_vocab_size: usize,
    classification: bool,
    max_seq_len: usize,
    train_count: usize, 
    test_count: usize,
    noise: bool,
    noise_map: &HashMap<usize, f64>,
    bootstrapping_iteration: usize,
    final_atom_id_list: Vec<u64>,
) -> io::Result<()> {
    let mut fps_test: Vec<Fingerprint> = Vec::new();
    let mut y_fixed_test: Vec<f64> = Vec::new();
    let mut fps_train: Vec<Fingerprint> = Vec::new();
    let mut y_train: Vec<f64> = Vec::new();
    let tokenizer = SmilesTokenizer::new();

    let train_file_path = format!("train_{}.mmap", bootstrapping_iteration);
    let test_file_path = format!("test_{}.mmap", bootstrapping_iteration);

    let train_file = File::open(&train_file_path)?;
    let test_file = File::open(&test_file_path)?;

    let mut train_reader = BufReader::new(train_file);
    train_reader.seek(SeekFrom::Start(0))?;
    for index in 0..train_count {
        if let Some(smiles_data) = read_smiles_data(&mut train_reader) {
            let mut property_value = smiles_data.qm_property_value;
            if noise {
                if let Some(&artificial_noise) = noise_map.get(&index) {
                    property_value += artificial_noise;
                }
            }
            let y_normalized = (property_value - mean) / std_dev;

            let fp_option: Option<Fingerprint> = match molecular_representation {
                "ecfp4" => {
                    let_cxx_string!(smiles_cxx = smiles_data.isomeric_smiles);
                    match smiles_to_mol(&smiles_cxx) {
                        Ok(mol) => {
                            let fingerprint = fingerprint_mol(&mol);
                            let cxx_vec_ptr: UniquePtr<CxxVector<u64>> = explicit_bit_vect_to_u64_vec(&fingerprint);
                            let cxx_vec_ref: &CxxVector<u64> = &*cxx_vec_ptr;
                            let u64_vec: Vec<u64> = cxx_vec_ref.iter().copied().collect();
                            Some(Fingerprint::ECFP4(u64_vec))
                        },
                        Err(_) => {
                            continue;
                        }
                    }
                },
                "smiles" => {
                    let smiles_ohe = smiles_to_ohe(&smiles_data.canonical_smiles, &tokenizer, &vocab, vocab_size, max_seq_len);
                    Some(Fingerprint::SMILES(smiles_ohe))
                },
                "alternative_smiles" => {
                    let smiles_ohe = smiles_to_ohe(&smiles_data.alternative_smiles, &tokenizer, &vocab, vocab_size, max_seq_len);
                    Some(Fingerprint::SMILES(smiles_ohe))
                }
                "sns" => {
                    let substructures = ecfp_atom_ids_from_smiles(&smiles_data.canonical_smiles).unwrap();
                    let mut sns_fp: Vec<u64> = vec![0; (substructure_vocab_size + 63) / 64];
                    for sub in substructures.keys() {
                        if let Some(&index) = substructure_vocab.get(sub) {
                            if final_atom_id_list.contains(sub) { // Check if the atom ID is in the final list
                                if index < substructure_vocab_size {
                                    sns_fp[index / 64] |= 1 << (index % 64);
                                }
                            }
                        }
                    }
                    Some(Fingerprint::ECFP4(sns_fp))
                },
                "mpnn" | "smiles_raw" => {
                    Some(Fingerprint::SRAW(smiles_data.canonical_smiles.clone()))
                },

                _ => None,
            };

            if let Some(fp) = fp_option {
                fps_train.push(fp);
                y_train.push(y_normalized);
            }
        }
    }

    let mut test_reader = BufReader::new(test_file);
    test_reader.seek(SeekFrom::Start(0))?;
    for _ in 0..test_count {
        if let Some(smiles_data) = read_smiles_data(&mut test_reader) {
            let mut property_value = smiles_data.qm_property_value;
            let y_normalized = (property_value - mean) / std_dev;

            let fp_option: Option<Fingerprint> = match molecular_representation {
                "ecfp4" => {
                    let_cxx_string!(smiles_cxx = smiles_data.isomeric_smiles);
                    match smiles_to_mol(&smiles_cxx) {
                        Ok(mol) => {
                            let fingerprint = fingerprint_mol(&mol);
                            let cxx_vec_ptr: UniquePtr<CxxVector<u64>> = explicit_bit_vect_to_u64_vec(&fingerprint);
                            let cxx_vec_ref: &CxxVector<u64> = &*cxx_vec_ptr;
                            let u64_vec: Vec<u64> = cxx_vec_ref.iter().copied().collect();
                            Some(Fingerprint::ECFP4(u64_vec))
                        },
                        Err(_) => {
                            continue;
                        }
                    }
                },
                "smiles" => {
                    let smiles_ohe = smiles_to_ohe(&smiles_data.canonical_smiles, &tokenizer, &vocab, vocab_size, max_seq_len);
                    Some(Fingerprint::SMILES(smiles_ohe))
                },
                "alternative_smiles" => {
                    let smiles_ohe = smiles_to_ohe(&smiles_data.alternative_smiles, &tokenizer, &vocab, vocab_size, max_seq_len);
                    Some(Fingerprint::SMILES(smiles_ohe))
                }
                "sns" => {
                    let substructures = ecfp_atom_ids_from_smiles(&smiles_data.canonical_smiles).unwrap();
                    let mut sns_fp: Vec<u64> = vec![0; (substructure_vocab_size + 63) / 64];
                    for sub in substructures.keys() {
                        if let Some(&index) = substructure_vocab.get(sub) {
                            if index < substructure_vocab_size {
                                sns_fp[index / 64] |= 1 << (index % 64);
                            }
                        }
                    }
                    Some(Fingerprint::ECFP4(sns_fp))
                },
                "mpnn" | "smiles_raw" => {
                    Some(Fingerprint::SRAW(smiles_data.canonical_smiles.clone()))
                },
                _ => None,
            };

            if let Some(fp) = fp_option {
                fps_test.push(fp);
                y_fixed_test.push(y_normalized);
            }
        }
    }

    // println!("fps_train length: {}", fps_train.len());
    // println!("y_train length: {}", y_train.len());
    // println!("fps_test length: {}", fps_test.len());
    // println!("y_fixed_test length: {}", y_fixed_test.len());

    send_to_python(fps_train.clone(), y_train.clone(), fps_test.clone(), y_fixed_test.clone());

    // TODO: Uncomment this to add back in ML in rust

    // if !fps_train.is_empty() {
    //     print_sample_fingerprints(&fps_train, "Training");
    // }
    // // if !fps_test.is_empty() {
    // //     print_sample_fingerprints(&fps_test, "Testing");
    // // }
    // // Tensorflow test 
    // // let values: Vec<u64> = (0..100000).collect();
    // // let t = Tensor::new(&[2, 50000]).with_values(&values).unwrap();
    // // dbg!(t);

    // // R-squared value to be collected and sent to Python
    // let mut r_squared_value = 0.0;

    // let start = Instant::now();
    // match molecular_representation {
    //     "ecfp4" => {
    //         if model == "gb" {
    //             let train_dmatrix = ecfp4_to_dmatrix(fps_train, y_train)
    //                 .expect("Failed to create training DMatrix");
    //             let test_dmatrix = ecfp4_to_dmatrix(fps_test, y_fixed_test)
    //                 .expect("Failed to create testing DMatrix");

    //             // Print details about the DMatrix objects
    //             // println!("Train DMatrix - Rows: {}, Columns: {}", train_dmatrix.num_rows(), train_dmatrix.num_cols());
    //             // println!("Test DMatrix - Rows: {}, Columns: {}", test_dmatrix.num_rows(), test_dmatrix.num_cols());

    //             // Assuming we can access labels directly or through a method
    //             let train_labels = train_dmatrix.get_labels().expect("Failed to get train labels");
    //             let test_labels = test_dmatrix.get_labels().expect("Failed to get test labels");
    //             // println!("Average train labels: {}", train_labels.iter().sum::<f32>() / train_labels.len() as f32);
    //             // println!("Average test labels: {}", test_labels.iter().sum::<f32>() / test_labels.len() as f32);

    //             let params = vec![("max_depth", "5"), ("eta", "0.1"), ("silent", "1"), ("objective", "binary:logistic")];
    //             let num_boost_round = 10;

    //             let (predictions, r2) = train_and_predict_with_xgboost(train_dmatrix, test_dmatrix, num_boost_round)
    //                 .expect("Model training or prediction failed");
    //             r_squared_value = r2; // Capture the R-squared value
    //         }
    //     },
    //     "smiles" => {
    //         if model == "gb" {
    //             // Attempt to create the training and testing DMatrix instances
    //             let train_dmatrix = smiles_to_dmatrix(fps_train, y_train, vocab_size);
    //             let test_dmatrix = smiles_to_dmatrix(fps_test, y_fixed_test, vocab_size);

    //             // Configuration parameters for the XGBoost model
    //             let params = vec![
    //                 ("max_depth", "5"),
    //                 ("eta", "0.1"),
    //                 ("silent", "1"),
    //                 ("objective", "binary:logistic") 
    //             ];
    //             let num_boost_round = 10;

    //             // Train the model and predict using the DMatrix instances
    //             let (predictions, r2) = train_and_predict_with_xgboost(train_dmatrix, test_dmatrix, num_boost_round)
    //                 .expect("Model training or prediction failed");
    //             r_squared_value = r2; // Capture the R-squared value

    //             // Output predictions or handle them as needed
    //             // println!("Predictions: {:?}", predictions);
    //         }
    //     },
    //     _ => {
    //         panic!("Unsupported molecular representation: {}", molecular_representation);
    //     }
    // }
    // let duration = start.elapsed();
    // // println!("Time elapsed in model is: {:?}", duration);

    // println!("{}", r_squared_value.to_string());

    Ok(())
}

fn main() -> io::Result<()> {
    // env::set_var("DYLD_LIBRARY_PATH", "/usr/local/Cellar/libtensorflow/2.15.0");

    let app = Command::new("My Rust Processor")
        .arg(Arg::new("seed")
             .long("seed")
             .action(ArgAction::Set)
             .help("Random seed for the process"))
        .arg(Arg::new("molecular_representation")
             .long("molecular_representation")
             .action(ArgAction::Set)
             .help("Molecular representation"))
        .arg(Arg::new("model")
             .long("model")
             .action(ArgAction::Set)
             .help("Model to use for prediction"))
        .arg(Arg::new("sigma")
             .long("sigma")
             .action(ArgAction::Set)
             .help("Sigma for artificial noise addition"))
        .arg(Arg::new("sampling_proportion")
             .long("sampling_proportion")
             .action(ArgAction::Set)
             .help("Sampling proportion for artificial noise addition"))
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
    let molecular_representation = matches.get_one::<String>("molecular_representation").unwrap();
    let model = matches.get_one::<String>("model").unwrap();
    let sigma: f64 = matches.get_one::<String>("sigma").unwrap().parse().expect("Sigma must be a valid float");
    let sampling_proportion: f64 = matches.get_one::<String>("sampling_proportion").unwrap().parse().expect("Sampling proportion must be a valid float");
    // let noise_mu: f64 = matches.get_one::<String>("noise_mu").unwrap().parse().expect("Noise mu must be a valid float");
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
    // let alpha_skew: f64 = matches.get_one::<String>("alpha_skew").unwrap().parse().expect("Alpha skew must be a valid float");
    // let beta_param1: f64 = matches.get_one::<String>("beta_param1").unwrap().parse().expect("Beta param1 must be a valid float");
    // let beta_param2: f64 = matches.get_one::<String>("beta_param2").unwrap().parse().expect("Beta param2 must be a valid float");
    let alpha_skew: f64 = 0.0;
    let beta_param1: f64 = 0.5;
    let beta_param2: f64 = 0.5;
    let noise_mu: f64 = 0.0;

    // Reading the configuration file
    let config_file = File::open("config.json")?;
    let reader = BufReader::new(config_file);
    let config: Config = serde_json::from_reader(reader)
                          .expect("JSON was not well-formatted or did not match the expected structure");

    // Iniitalise the Python interpreter for SNS
    if molecular_representation == "sns" {
        pyo3::prepare_freethreaded_python();
    }

    let noise_indices: Vec<usize> = if config.noise {
        (0..config.train_count)
            .filter(|_| rand::random::<f64>() < sampling_proportion)
            .collect()
    } else {
        Vec::new()
    };

    // if domain_noise {
    //     let mut fps_train: Vec<Fingerprint> = Vec::new();
    //     let tokenizer = SmilesTokenizer::new();
    //     let mut train_reader = BufReader::new(train_file);
    //     train_reader.seek(SeekFrom::Start(0))?;
    //     for _ in 0..train_count {
    //         if let Some(smiles_data) = read_smiles_data(&mut train_reader) {
    //             let fp_option: Option<Fingerprint> = match molecular_representation {
    //                 "ecfp4" => {
    //                     let_cxx_string!(smiles_cxx = smiles_data.isomeric_smiles);
    //                     match smiles_to_mol(&smiles_cxx) {
    //                         Ok(mol) => {
    //                             let fingerprint = fingerprint_mol(&mol);
    //                             let cxx_vec_ptr: UniquePtr<CxxVector<u64>> = explicit_bit_vect_to_u64_vec(&fingerprint);
    //                             let cxx_vec_ref: &CxxVector<u64> = &*cxx_vec_ptr;
    //                             let u64_vec: Vec<u64> = cxx_vec_ref.iter().copied().collect();
    //                             Some(Fingerprint::ECFP4(u64_vec))
    //                         },
    //                         Err(_) => {
    //                             continue;
    //                         }
    //                     }
    //                 },
    //                 "smiles" => {
    //                     let smiles_ohe = smiles_to_ohe(&smiles_data.canonical_smiles, &tokenizer, &vocab, vocab_size, max_seq_len);
    //                     Some(Fingerprint::SMILES(smiles_ohe))
    //                 },
    //                 _ => None,
    //             };

    //             if let Some(fp) = fp_option {
    //                 fps_train.push(fp);
    //             }
    //         }
    //     }

    //     let ecfp4_vectors: Vec<Vec<f64>> = fps_train.iter().filter_map(|fp| {
    //         if let Fingerprint::ECFP4(bits) = fp {
    //             Some(bits.iter().flat_map(|&num| {
    //                 (0..64).map(move |i| {
    //                     if num & (1 << i) != 0 { 1.0 } else { 0.0 }
    //                 })
    //             }).collect())
    //         } else {
    //             None
    //         }
    //     }).collect();

    //     let n_clusters = 5; // Example number of clusters, adjust as needed
    //     let dataset = Dataset::new(ecfp4_vectors);
    //     let kmeans = LinfaKMeans::params(n_clusters)
    //         .max_n_iterations(100)
    //         .fit(&dataset)
    //         .expect("Failed to fit k-means");

    //     let cluster_assignments = kmeans.predict(&dataset);

    //     let cluster_to_add_noise = 0; // Example cluster to add noise to, adjust as needed
    //     let noise_indices: Vec<usize> = cluster_assignments
    //         .iter()
    //         .enumerate()
    //         .filter_map(|(idx, &cluster)| if cluster == cluster_to_add_noise { Some(idx) } else { None })
    //         .collect();
    // }

    // else {

    let noise_map: HashMap<usize, f64> = generate_noise_by_indices(
        &noise_indices,
        noise_mu,
        sigma,
        noise_distribution.clone(),
        alpha_skew,
        (beta_param1, beta_param2),
        seed,
    );

    let (mean, std_dev, vocab_size, vocab, max_sequence_length, substructure_vocab, substructure_vocab_size, final_atom_id_list) = generate_aggregate_stats(
        seed,
        molecular_representation,
        config.train_count,
        config.sample_size,
        config.noise,
        &noise_map,
        config.max_vocab,
        config.bootstrapping_iteration,
        config.target_domain,
        noise_distribution,
    )?;

    let classification: bool = false;

    // println!("Starting data preprocessing...");
    preprocess_data(
        seed,
        molecular_representation,
        model,
        config.sample_size,
        mean,
        std_dev,
        vocab_size,
        &vocab,
        &substructure_vocab,
        substructure_vocab_size,
        classification,
        max_sequence_length,
        config.train_count, 
        config.test_count,
        config.noise,
        &noise_map,
        config.bootstrapping_iteration,
        final_atom_id_list,
    )?;
    // println!("Data preprocessing complete");

    Ok(())
}
