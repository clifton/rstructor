use rstructor::Instructor;

#[derive(Instructor)]
#[llm(validate = "risk::")]
struct InvalidValidationPath {
    value: String,
}

fn main() {}
