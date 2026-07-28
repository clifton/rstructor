use rstructor::Instructor;

#[derive(Instructor)]
#[llm(examples = "not an array")]
struct InvalidContainerExamples {
    value: String,
}

#[derive(Instructor)]
struct InvalidFieldExamples {
    #[llm(examples = 42)]
    value: String,
}

fn main() {}
