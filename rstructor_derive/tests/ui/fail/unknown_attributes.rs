use rstructor::Instructor;

#[derive(Instructor)]
#[llm(descripton = "misspelled")]
struct UnknownContainer {
    #[llm(exampel = 42)]
    value: i32,

    #[llm(optional)]
    legacy_optional: Option<String>,
}

#[derive(Instructor)]
enum UnknownVariant {
    #[llm(label = "wrong namespace key")]
    Active,
}

fn main() {}
