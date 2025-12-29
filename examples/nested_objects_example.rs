#![allow(clippy::collapsible_if)]

use rstructor::{AnthropicClient, Instructor, LLMClient, OpenAIClient, RStructorError};
use serde::{Deserialize, Serialize};
use std::env;

// Define a nested data model for a recipe
#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(description = "A cooking ingredient with amount and unit")]
struct Ingredient {
    #[llm(description = "Name of the ingredient", example = "flour")]
    name: String,

    #[llm(description = "Amount of the ingredient", example = 2.5)]
    amount: f32,

    #[llm(description = "Unit of measurement", example = "cups")]
    unit: String,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(description = "A step in the recipe instructions")]
struct Step {
    #[llm(description = "Order number of this step", example = 1)]
    number: u16,

    #[llm(
        description = "Description of this step",
        example = "Mix the flour and sugar together"
    )]
    description: String,

    #[llm(description = "Estimated time for this step in minutes", example = 5)]
    time_minutes: Option<u16>,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(description = "Nutritional information per serving. All values are numbers, not strings.")]
struct Nutrition {
    #[llm(
        description = "Calories per serving (must be a number, not a string)",
        example = 350
    )]
    calories: u16,

    #[llm(
        description = "Protein in grams (must be a number, field name is 'protein_g')",
        example = 7.5
    )]
    protein_g: f32,

    #[llm(
        description = "Carbohydrates in grams (must be a number, field name is 'carbs_g', not 'carbohydrates')",
        example = 45.0
    )]
    carbs_g: f32,

    #[llm(
        description = "Fat in grams (must be a number, field name is 'fat_g')",
        example = 15.2
    )]
    fat_g: f32,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(description = "A cooking recipe with ingredients and instructions",
      validate = "validate_recipe",
      examples = [
        ::serde_json::json!({
            "name": "Chocolate Chip Cookies",
            "description": "Classic homemade chocolate chip cookies that are soft and chewy.",
            "prep_time_minutes": 15,
            "cook_time_minutes": 12,
            "servings": 24,
            "difficulty": "Easy",
            "ingredients": [
                {"name": "all-purpose flour", "amount": 2.25, "unit": "cups"},
                {"name": "baking soda", "amount": 1, "unit": "teaspoon"},
                {"name": "salt", "amount": 1, "unit": "teaspoon"},
                {"name": "butter", "amount": 1, "unit": "cup"}
            ],
            "steps": [
                {"number": 1, "description": "Preheat oven to 375°F.", "time_minutes": 2},
                {"number": 2, "description": "Mix flour, baking soda, and salt in a bowl.", "time_minutes": 3}
            ],
            "nutrition": {
                "calories": 150,
                "protein_g": 2.0,
                "carbs_g": 18.5,
                "fat_g": 7.8
            }
        })
      ])]
struct Recipe {
    #[llm(description = "Name of the recipe", example = "Banana Bread")]
    name: String,

    #[llm(
        description = "Short description of the recipe",
        example = "Delicious homemade banana bread with walnuts"
    )]
    description: String,

    #[llm(description = "Preparation time in minutes", example = 20)]
    prep_time_minutes: u16,

    #[llm(description = "Cooking time in minutes", example = 60)]
    cook_time_minutes: u16,

    #[llm(description = "Number of servings this recipe makes", example = 8)]
    servings: u8,

    #[llm(description = "Recipe difficulty level", example = "Medium")]
    difficulty: String,

    #[llm(
        description = "List of ingredients needed. MUST be an array of objects, not strings. Each object must have 'name', 'amount', and 'unit' fields."
    )]
    ingredients: Vec<Ingredient>,

    #[llm(
        description = "Step-by-step cooking instructions. MUST be an array of objects, not strings. Each object must have 'number', 'description', and optionally 'time_minutes' fields."
    )]
    steps: Vec<Step>,

    #[llm(
        description = "Nutritional information per serving. MUST be an object with exactly these fields: calories (number), protein_g (number), carbs_g (number), fat_g (number). Field names must match exactly."
    )]
    nutrition: Nutrition,
}

// Custom validation function referenced by #[llm(validate = "validate_recipe")]
fn validate_recipe(recipe: &Recipe) -> rstructor::Result<()> {
    // Check that we have at least one ingredient
    if recipe.ingredients.is_empty() {
        return Err(RStructorError::ValidationError(
            "Recipe must have at least one ingredient".to_string(),
        ));
    }

    // Check that we have at least one step
    if recipe.steps.is_empty() {
        return Err(RStructorError::ValidationError(
            "Recipe must have at least one step".to_string(),
        ));
    }

    // Check that steps are numbered correctly (1-based, sequential)
    for (i, step) in recipe.steps.iter().enumerate() {
        if step.number != (i + 1) as u16 {
            return Err(RStructorError::ValidationError(format!(
                "Step numbers must be sequential, expected {} but got {}",
                i + 1,
                step.number
            )));
        }
    }

    // Check that difficulty is one of the expected values
    let valid_difficulties = vec!["Easy", "Medium", "Hard"];
    if !valid_difficulties.contains(&recipe.difficulty.as_str()) {
        return Err(RStructorError::ValidationError(format!(
            "Difficulty must be one of {:?}, got {}",
            valid_difficulties, recipe.difficulty
        )));
    }

    Ok(())
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // User prompt requesting a recipe
    // Important: Explicitly request structured data format
    let prompt = "Create a recipe for chocolate chip cookies.

CRITICAL REQUIREMENTS - ALL FIELDS ARE REQUIRED:
1. Ingredients MUST be an array of objects (not strings). Each object must have exactly: 'name' (string), 'amount' (number), 'unit' (string).
2. Steps MUST be an array of objects (not strings). Each object must have: 'number' (integer starting at 1), 'description' (string), and optionally 'time_minutes' (integer).
3. Nutrition MUST be an object with exactly these fields: 'calories' (integer), 'protein_g' (number), 'carbs_g' (number), 'fat_g' (number). All values must be numbers, not strings. Field names must match exactly. THIS FIELD IS REQUIRED - DO NOT OMIT IT.
4. All other fields (name, description, prep_time_minutes, cook_time_minutes, servings, difficulty) are also REQUIRED.";

    // Try using either OpenAI or Anthropic based on available API keys
    if let Ok(api_key) = env::var("OPENAI_API_KEY") {
        println!("Using OpenAI to generate recipe...");

        let client = OpenAIClient::new(api_key)?
            .temperature(0.2)
            .max_retries(5)
            .include_error_feedback(true);

        let recipe: Recipe = client.materialize(prompt).await?;

        // Print the generated recipe
        print_recipe(&recipe);
    } else if let Ok(api_key) = env::var("ANTHROPIC_API_KEY") {
        println!("Using Anthropic to generate recipe...");

        let client = AnthropicClient::new(api_key)?
            .temperature(0.2)
            .max_retries(5)
            .include_error_feedback(true);

        let recipe: Recipe = client.materialize(prompt).await?;

        // Print the generated recipe
        print_recipe(&recipe);
    } else {
        println!("No API keys found in environment variables.");
        println!("Please set either OPENAI_API_KEY or ANTHROPIC_API_KEY to run this example.");
        println!("\nNote: This example requires API keys because it makes actual LLM calls.");
        println!(
            "If you see validation errors, the LLM may need more retries or a clearer prompt."
        );
    }

    Ok(())
}

// Helper function to print the recipe nicely
fn print_recipe(recipe: &Recipe) {
    println!("\n===== {} =====", recipe.name);
    println!("{}\n", recipe.description);

    println!("Prep Time: {} minutes", recipe.prep_time_minutes);
    println!("Cook Time: {} minutes", recipe.cook_time_minutes);
    println!("Servings: {}", recipe.servings);
    println!("Difficulty: {}\n", recipe.difficulty);

    println!("--- Ingredients ---");
    for ingredient in &recipe.ingredients {
        println!(
            "• {} {} {}",
            ingredient.amount, ingredient.unit, ingredient.name
        );
    }

    println!("\n--- Instructions ---");
    for step in &recipe.steps {
        let time_info = if let Some(time) = step.time_minutes {
            format!(" ({} minutes)", time)
        } else {
            String::new()
        };

        println!("{}. {}{}", step.number, step.description, time_info);
    }

    println!("\n--- Nutrition (per serving) ---");
    println!("Calories: {}", recipe.nutrition.calories);
    println!("Protein: {}g", recipe.nutrition.protein_g);
    println!("Carbs: {}g", recipe.nutrition.carbs_g);
    println!("Fat: {}g", recipe.nutrition.fat_g);
}
