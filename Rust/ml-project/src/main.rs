use ndarray::{Array1, Array2};
use std::error::Error;
use csv::ReaderBuilder;
use linfa::prelude::*;
use linfa::dataset::DatasetBase;
use linfa_logistic::LogisticRegression;

fn load_data(file_path: &str) -> Result<DatasetBase<Array2<f64>, Array1<usize>>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(file_path)?;
    let mut features = Vec::new();
    let mut targets = Vec::new();
    for record in reader.records() {
        let record = record?;
        let target = if record[1] == *"M" { 1 } else { 0 };
        targets.push(target);
        let row: Vec<f64> = record.iter()
            .skip(2)
            .filter(|v| !v.is_empty())
            .map(|v: &str| v.parse::<f64>().unwrap_or(0.0))
            .collect();
        features.extend(row);
    }

    let num_features = 30;
    let num_rows = targets.len();
    let features = Array2::from_shape_vec((num_rows, num_features), features)?;
    let targets = Array1::from(targets);

    Ok(Dataset::new(features, targets))
}

fn evaluate_model(model: &linfa_logistic::FittedLogisticRegression<f64, usize>, dataset: &DatasetBase<Array2<f64>, Array1<usize>>) -> Result<(), Box<dyn Error>> {
    let y_pred = model.predict(dataset);
    let y_true = dataset.targets();
    let accuracy = y_pred.iter()
        .zip(y_true.iter())
        .filter(|(p, t)| p == t)
        .count() as f64 / y_true.len() as f64;
    println!("Model Accuracy: {:.2}%", accuracy * 100.0);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = load_data("data/breast_cancer.csv")?;
    let (train, test) = dataset.split_with_ratio(0.7);
    let model = LogisticRegression::default().fit(&train)?;
    evaluate_model(&model, &test)?;
    Ok(())
}