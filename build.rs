pub fn main() {
    println!("cargo::rerun-if-changed=src/feature_engineering/test_data");
    println!(
        "cargo::rerun-if-changed=src/rules_engine/test_data/processed_replays"
    );
}
