use rstructor::Instructor;

#[derive(Instructor)]
#[llm(description = 42)]
#[llm(title)]
struct InvalidContainerValues {
    value: String,
}

#[derive(Instructor)]
struct InvalidFieldExample {
    #[llm(example = false)]
    value: i64,

    #[llm(example = "42")]
    quoted_integer: i64,

    #[llm(example = "3.5")]
    quoted_float: f64,

    #[llm(example = "true")]
    quoted_boolean: bool,
}

fn main() {}
