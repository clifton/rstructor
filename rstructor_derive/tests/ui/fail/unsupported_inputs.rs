use rstructor::Instructor;

#[derive(Instructor)]
struct TupleStruct(String);

#[derive(Instructor)]
struct UnitStruct;

#[derive(Instructor)]
union UnsupportedUnion {
    integer: i64,
    float: f64,
}

fn main() {}
