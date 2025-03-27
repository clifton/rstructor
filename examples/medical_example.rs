use rstructor::{Instructor, SchemaType};
use serde::{Deserialize, Serialize};

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct Symptoms {
    #[llm(description = "A general scientific and objective description of the symptoms")]
    description: String,

    #[llm(
        description = "Type of pain experienced by the patient",
        example = "Sharp"
    )]
    pain_type: String,

    #[llm(description = "Areas of the body where symptoms occur", 
          example = ["Chest", "Left arm", "Jaw"])]
    locations: Vec<String>,

    #[llm(description = "Pain intensity on a scale of 1-10", example = 7)]
    intensity: u8,

    #[llm(
        description = "How precisely the patient can locate the pain on a scale of 1-10",
        example = 8
    )]
    location_precision: u8,

    #[llm(
        description = "Rate at which symptoms progress on a scale of 1-10",
        example = 5
    )]
    pace: u8,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct MedicalHistory {
    #[llm(
        description = "Known pathology or previously diagnosed condition",
        example = "Hypertension"
    )]
    pathology: String,

    #[llm(description = "Detailed symptoms information")]
    symptoms: Symptoms,

    #[llm(
        description = "Whether symptoms become worse with physical exertion",
        example = true
    )]
    increase_with_exertion: bool,

    #[llm(description = "Whether symptoms improve when resting", example = true)]
    alleviate_with_rest: bool,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct RiskFactors {
    #[llm(
        description = "Whether the condition appeared spontaneously",
        example = false
    )]
    spontaneous_history: bool,

    #[llm(
        description = "Whether the patient has a history of smoking",
        example = true
    )]
    smoking_history: bool,

    #[llm(
        description = "Whether the patient has a history of COPD",
        example = false
    )]
    copd_history: bool,

    #[llm(description = "Family history of related conditions", 
          examples = ["Father with coronary artery disease", "Mother with hypertension"])]
    family_history: String,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct DifferentialDiagnosis {
    #[llm(description = "Name of the possible disease", 
          examples = ["Myocardial infarction", "Angina pectoris", "Aortic dissection"])]
    disease_name: String,

    #[llm(description = "Probability of this diagnosis as a percentage", 
          examples = [0.75, 0.45, 0.25])]
    probability: f32,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
enum Sex {
    M,
    F,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct PatientInfo {
    #[llm(description = "Patient's biological sex")]
    sex: Sex,

    #[llm(description = "Patient's age in years", example = 65)]
    age: u8,

    #[llm(
        description = "Patient's geographical region",
        example = "North America"
    )]
    geographical_region: String,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct PatientData {
    #[llm(description = "Basic patient demographic information")]
    patient_info: PatientInfo,

    #[llm(description = "Patient's medical history and current symptoms")]
    medical_history: MedicalHistory,

    #[llm(description = "Patient's risk factors for various conditions")]
    risk_factors: RiskFactors,

    #[llm(description = "List of possible diagnoses with probabilities", 
          example = [
              {
                  "disease_name": "Myocardial infarction",
                  "probability": 0.85
              },
              {
                  "disease_name": "Angina pectoris",
                  "probability": 0.35
              }
          ])]
    differential_diagnosis: Vec<DifferentialDiagnosis>,
}

fn main() {
    // Get the schema for the patient data
    let schema = PatientData::schema();

    println!("Medical Diagnostic Schema:");
    println!(
        "{}",
        serde_json::to_string_pretty(&schema.to_json()).unwrap()
    );

    // Create a sample patient case
    let patient = PatientData {
        patient_info: PatientInfo {
            sex: Sex::M,
            age: 65,
            geographical_region: "North America".to_string(),
        },
        medical_history: MedicalHistory {
            pathology: "Hypertension".to_string(),
            symptoms: Symptoms {
                description: "Patient presents with crushing chest pain radiating to left arm"
                    .to_string(),
                pain_type: "Crushing".to_string(),
                locations: vec!["Chest".to_string(), "Left arm".to_string()],
                intensity: 8,
                location_precision: 7,
                pace: 6,
            },
            increase_with_exertion: true,
            alleviate_with_rest: true,
        },
        risk_factors: RiskFactors {
            spontaneous_history: false,
            smoking_history: true,
            copd_history: false,
            family_history: "Father with coronary artery disease".to_string(),
        },
        differential_diagnosis: vec![
            DifferentialDiagnosis {
                disease_name: "Myocardial infarction".to_string(),
                probability: 0.85,
            },
            DifferentialDiagnosis {
                disease_name: "Angina pectoris".to_string(),
                probability: 0.35,
            },
            DifferentialDiagnosis {
                disease_name: "Aortic dissection".to_string(),
                probability: 0.15,
            },
        ],
    };

    // Serialize to JSON
    println!("\nSample Patient Data:");
    println!("{}", serde_json::to_string_pretty(&patient).unwrap());
}
