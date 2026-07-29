use std::fs;
use std::path::{Path, PathBuf};
use std::{collections::BTreeSet, ffi::OsStr};

const GALLERY_EXAMPLES: &[(&str, &[&str])] = &[
    ("structured_movie_info", &["derive", "openai"]),
    ("news_article_categorizer", &["derive", "grok"]),
    ("openai_multimodal_example", &["derive", "openai"]),
    ("anthropic_multimodal_example", &["anthropic", "derive"]),
    ("gemini_multimodal_example", &["derive", "gemini"]),
    ("grok_multimodal_example", &["derive", "grok"]),
    ("axum_handler_example", &["derive", "mock"]),
    ("mock_testing_example", &["derive", "mock"]),
    ("ollama_local_example", &["derive", "openai"]),
    ("runtime_provider_example", &["derive", "openai"]),
    ("schemars_bridge_example", &["mock", "schemars"]),
    ("tool_calling_example", &["derive", "openai", "tools"]),
    ("streaming_example", &["derive", "openai", "streaming"]),
    ("validation_example", &["derive"]),
    ("retry_attempt_ledger", &["derive", "openai"]),
    ("nested_objects_example", &["derive", "gemini"]),
    ("recursive_schema_graph", &["derive"]),
];

const OTHER_FEATURE_GATED_EXAMPLES: &[(&str, &[&str])] = &[
    ("container_attributes_example", &["derive"]),
    ("custom_type_example", &["derive"]),
    ("enum_example", &["derive"]),
    ("enum_with_data_example", &["derive"]),
    ("event_planner", &["anthropic", "derive"]),
    ("kimi_k3_multimodal_example", &["derive", "openai"]),
    ("logging_example", &["anthropic", "derive", "logging"]),
    ("medical_example", &["derive"]),
    ("movie_example", &["derive"]),
    ("nested_enum_example", &["derive"]),
    ("serde_rename_example", &["derive"]),
    ("token_usage_example", &["derive", "openai"]),
    ("weather_example", &["derive", "openai"]),
];

const COOKBOOK_RECIPES: &[(&str, &str)] = &[
    (
        "## Extract typed data from an image or PDF",
        "openai_multimodal_example",
    ),
    ("## Classify text into an enum", "news_article_categorizer"),
    (
        "## Put typed extraction behind an axum handler",
        "axum_handler_example",
    ),
    (
        "## Test extraction offline with `MockClient`",
        "mock_testing_example",
    ),
    (
        "## Use a local model through Ollama",
        "ollama_local_example",
    ),
    (
        "## Choose a provider at runtime",
        "runtime_provider_example",
    ),
    ("## Reuse a schemars model", "schemars_bridge_example"),
    ("## Inspect cumulative retry cost", "retry_attempt_ledger"),
];

fn repo_path(path: impl AsRef<Path>) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(path)
}

fn read(path: impl AsRef<Path>) -> String {
    let path = repo_path(path);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

fn example_manifest_block<'a>(manifest: &'a str, example: &str) -> &'a str {
    manifest
        .split("[[example]]")
        .find(|block| block.contains(&format!("name = \"{example}\"")))
        .unwrap_or_else(|| panic!("Cargo.toml must declare `{example}`"))
}

#[test]
fn every_gallery_example_exists_and_is_linked() {
    let readme = read("README.md");

    for (example, _) in GALLERY_EXAMPLES {
        let source = format!("examples/{example}.rs");
        assert!(
            repo_path(&source).is_file(),
            "gallery target `{source}` must exist"
        );
        assert!(
            readme.contains(&format!("]({source})")),
            "README gallery must link `{source}`"
        );
    }
}

#[test]
fn every_feature_dependent_example_declares_its_required_features() {
    let manifest = read("Cargo.toml");

    for (example, features) in GALLERY_EXAMPLES.iter().chain(OTHER_FEATURE_GATED_EXAMPLES) {
        let block = example_manifest_block(&manifest, example);
        for feature in *features {
            assert!(
                block.contains(&format!("\"{feature}\"")),
                "`{example}` must require its `{feature}` feature: {block}"
            );
        }
    }
}

#[test]
fn feature_audit_covers_the_entire_examples_directory() {
    let audited = GALLERY_EXAMPLES
        .iter()
        .chain(OTHER_FEATURE_GATED_EXAMPLES)
        .map(|(example, _)| *example)
        .chain(["manual_implementation"])
        .collect::<BTreeSet<_>>();
    let present = fs::read_dir(repo_path("examples"))
        .expect("read examples directory")
        .map(|entry| entry.expect("read example directory entry").path())
        .filter(|path| path.extension() == Some(OsStr::new("rs")))
        .map(|path| {
            path.file_stem()
                .and_then(OsStr::to_str)
                .expect("example filename must be UTF-8")
                .to_string()
        })
        .collect::<BTreeSet<_>>();
    let audited = audited.into_iter().map(str::to_string).collect();

    assert_eq!(
        present, audited,
        "every runnable example must be covered by the feature-gating audit"
    );
}

#[test]
fn recipes_gallery_precedes_providers_and_links_the_cookbook() {
    let readme = read("README.md");
    let gallery = readme.find("## Recipes").expect("README recipes gallery");
    let providers = readme
        .find("## Providers")
        .expect("README providers section");

    assert!(gallery < providers, "recipes must appear before providers");
    assert!(
        readme.contains("[task-oriented cookbook](docs/COOKBOOK.md)"),
        "README must prominently link the cookbook"
    );
}

#[test]
fn cookbook_covers_every_requested_task_with_a_runnable_example() {
    let cookbook = read("docs/COOKBOOK.md");

    for (heading, example) in COOKBOOK_RECIPES {
        assert!(cookbook.contains(heading), "missing recipe `{heading}`");

        let link = format!("](../examples/{example}.rs)");
        assert!(
            cookbook.contains(&link),
            "recipe `{heading}` must link runnable example `{example}`"
        );
        assert!(
            repo_path(format!("examples/{example}.rs")).is_file(),
            "cookbook target `{example}` must exist"
        );
    }

    for line in cookbook.lines().filter(|line| line.starts_with("```")) {
        assert!(
            !line.contains("ignore"),
            "cookbook doctests must never be ignored: {line}"
        );
    }
}
