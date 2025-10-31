use rstructor::{
    AnthropicClient, AnthropicModel, Instructor, LLMClient, OpenAIClient, OpenAIModel,
    RStructorError,
};
use serde::{Deserialize, Serialize};
use std::{
    env,
    io::{self, Write},
};

// Define a nested data model for a recipe
#[derive(Instructor, Serialize, Deserialize, Debug)]
struct Ingredient {
    #[llm(description = "Name of the ingredient", example = "flour")]
    name: String,

    #[llm(
        description = "Numeric amount of the ingredient (e.g., 2.0, 0.5, etc.)",
        example = 2.5
    )]
    amount: f32,

    #[llm(description = "Unit of measurement", example = "cups")]
    unit: String,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
struct Step {
    #[llm(description = "Order number of this step", example = 1)]
    number: u16,

    #[llm(
        description = "Description of this step",
        example = "Mix the flour and sugar together"
    )]
    description: String,
}

#[derive(Instructor, Serialize, Deserialize, Debug)]
#[llm(description = "A complete cooking recipe with ingredients and step-by-step instructions",
      examples = [
        ::serde_json::json!({
            "name": "Classic Chocolate Chip Cookies",
            "ingredients": [
                {"name": "all-purpose flour", "amount": 2.25, "unit": "cups"},
                {"name": "baking soda", "amount": 1.0, "unit": "teaspoon"},
                {"name": "salt", "amount": 1.0, "unit": "teaspoon"},
                {"name": "butter", "amount": 1.0, "unit": "cup"},
                {"name": "granulated sugar", "amount": 0.75, "unit": "cup"},
                {"name": "brown sugar", "amount": 0.75, "unit": "cup"},
                {"name": "vanilla extract", "amount": 1.0, "unit": "teaspoon"},
                {"name": "eggs", "amount": 2.0, "unit": "large"},
                {"name": "chocolate chips", "amount": 2.0, "unit": "cups"}
            ],
            "steps": [
                {"number": 1, "description": "Preheat oven to 375°F (190°C)."},
                {"number": 2, "description": "In a small bowl, mix flour, baking soda, and salt."},
                {"number": 3, "description": "In a large bowl, cream together butter and both sugars until smooth."},
                {"number": 4, "description": "Beat in vanilla and eggs one at a time."},
                {"number": 5, "description": "Gradually blend in the dry ingredients."},
                {"number": 6, "description": "Stir in chocolate chips."},
                {"number": 7, "description": "Drop by rounded tablespoons onto ungreased cookie sheets."},
                {"number": 8, "description": "Bake for 9 to 11 minutes or until golden brown."},
                {"number": 9, "description": "Let stand for 2 minutes before removing to cool on wire racks."}
            ]
        })
      ])]
struct Recipe {
    #[llm(description = "Name of the recipe", example = "Chocolate Chip Cookies")]
    name: String,

    #[llm(description = "List of ingredients needed")]
    ingredients: Vec<Ingredient>,

    #[llm(description = "Step-by-step cooking instructions")]
    steps: Vec<Step>,
}

// Add custom validation
impl Recipe {
    fn validate(&self) -> rstructor::Result<()> {
        // Recipe must have a name
        if self.name.trim().is_empty() {
            return Err(RStructorError::ValidationError(
                "Recipe must have a name".to_string(),
            ));
        }

        // Must have at least one ingredient
        if self.ingredients.is_empty() {
            return Err(RStructorError::ValidationError(
                "Recipe must have at least one ingredient".to_string(),
            ));
        }

        // Must have at least one step
        if self.steps.is_empty() {
            return Err(RStructorError::ValidationError(
                "Recipe must have at least one step".to_string(),
            ));
        }

        // Validate steps are in order
        let mut prev_number = 0;
        for step in &self.steps {
            if step.number <= prev_number {
                return Err(RStructorError::ValidationError(format!(
                    "Step numbers must be sequential, found step {} after step {}",
                    step.number, prev_number
                )));
            }
            prev_number = step.number;
        }

        // All ingredients must have positive amounts
        for ingredient in &self.ingredients {
            if ingredient.amount <= 0.0 {
                return Err(RStructorError::ValidationError(format!(
                    "Ingredient '{}' has invalid amount: {}",
                    ingredient.name, ingredient.amount
                )));
            }

            // Ingredient name can't be empty
            if ingredient.name.trim().is_empty() {
                return Err(RStructorError::ValidationError(
                    "Ingredient name cannot be empty".to_string(),
                ));
            }

            // Unit can't be empty
            if ingredient.unit.trim().is_empty() {
                return Err(RStructorError::ValidationError(format!(
                    "Unit cannot be empty for ingredient '{}'",
                    ingredient.name
                )));
            }
        }

        Ok(())
    }
}

async fn get_recipe_from_openai(recipe_name: &str) -> rstructor::Result<Recipe> {
    // Get API key from environment
    let api_key = env::var("OPENAI_API_KEY").map_err(|_| {
        RStructorError::ApiError("OPENAI_API_KEY environment variable not set".into())
    })?;

    // Create OpenAI client
    let client = OpenAIClient::new(api_key)?
        .model(OpenAIModel::Gpt4O) // Use GPT-4o for better recipes
        .temperature(0.1) // Lower temperature for more consistent results
        .max_retries(3)
        .include_error_feedback(true);

    // Generate the recipe with a structured prompt
    let prompt = format!(
        "Create a recipe for {}. Your response must be valid, structured JSON with the following format:\n\
        {{\n\
          \"name\": \"Recipe Name\",\n\
          \"ingredients\": [\n\
            {{ \"name\": \"ingredient name\", \"amount\": 2.0, \"unit\": \"cups\" }},\n\
            {{ \"name\": \"another ingredient\", \"amount\": 1.0, \"unit\": \"tablespoon\" }}\n\
          ],\n\
          \"steps\": [\n\
            {{ \"number\": 1, \"description\": \"First step instruction\" }},\n\
            {{ \"number\": 2, \"description\": \"Second step instruction\" }}\n\
          ]\n\
        }}\n\n\
        IMPORTANT:\n\
        - Include at least 5 ingredients with proper measurements\n\
        - All numerical amounts must be decimal numbers (like 1.0, 2.5, not integers)\n\
        - Include at least 5 detailed steps\n\
        - Step numbers must be sequential starting with 1\n\
        - Return ONLY valid JSON with no additional explanation",
        recipe_name
    );

    // Use the library's built-in retry functionality
    client.generate_struct::<Recipe>(&prompt).await
}

async fn get_recipe_from_anthropic(recipe_name: &str) -> rstructor::Result<Recipe> {
    // Get API key from environment
    let api_key = env::var("ANTHROPIC_API_KEY").map_err(|_| {
        RStructorError::ApiError("ANTHROPIC_API_KEY environment variable not set".into())
    })?;

    // Create Anthropic client
    let client = AnthropicClient::new(api_key)?
        .model(AnthropicModel::ClaudeSonnet45) // Use Claude Sonnet 4.5 for better recipes
        .temperature(0.1) // Lower temperature for more consistent results
        .max_retries(3)
        .include_error_feedback(true);

    // Generate the recipe with a structured prompt
    let prompt = format!(
        "Create a recipe for {}. Your response must be valid, structured JSON with the following format:\n\
        {{\n\
          \"name\": \"Recipe Name\",\n\
          \"ingredients\": [\n\
            {{ \"name\": \"ingredient name\", \"amount\": 2.0, \"unit\": \"cups\" }},\n\
            {{ \"name\": \"another ingredient\", \"amount\": 1.0, \"unit\": \"tablespoon\" }}\n\
          ],\n\
          \"steps\": [\n\
            {{ \"number\": 1, \"description\": \"First step instruction\" }},\n\
            {{ \"number\": 2, \"description\": \"Second step instruction\" }}\n\
          ]\n\
        }}\n\n\
        IMPORTANT:\n\
        - Include at least 5 ingredients with proper measurements\n\
        - All numerical amounts must be decimal numbers (like 1.0, 2.5, not integers)\n\
        - Include at least 5 detailed steps\n\
        - Step numbers must be sequential starting with 1\n\
        - Return ONLY valid JSON with no additional explanation",
        recipe_name
    );

    // Use the library's built-in retry functionality
    client.generate_struct::<Recipe>(&prompt).await
}

fn print_recipe(recipe: &Recipe) {
    println!("\n{}", "=".repeat(50));
    println!("📝 {}", recipe.name);
    println!("{}", "=".repeat(50));

    println!("\n🧾 Ingredients:");
    println!("{}", "-".repeat(50));
    for ingredient in &recipe.ingredients {
        println!(
            "• {:.2} {} {}",
            ingredient.amount, ingredient.unit, ingredient.name
        );
    }

    println!("\n👨‍🍳 Instructions:");
    println!("{}", "-".repeat(50));
    for step in &recipe.steps {
        println!("{}. {}", step.number, step.description);
    }

    println!("\nEnjoy your {}! 🍽️\n", recipe.name);
}

// Helper function to handle recipe extraction
async fn run_recipe_extraction(recipe_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nFetching recipe for {}...", recipe_name);

    // Try OpenAI first, fall back to Anthropic if OpenAI key isn't set
    if env::var("OPENAI_API_KEY").is_ok() {
        match get_recipe_from_openai(recipe_name).await {
            Ok(recipe) => {
                println!("Recipe successfully generated with OpenAI! 🎉");
                print_recipe(&recipe);
            }
            Err(e) => {
                println!("Error getting recipe from OpenAI: {}", e);

                // Fallback to Anthropic if available
                if env::var("ANTHROPIC_API_KEY").is_ok() {
                    println!("Trying Anthropic as fallback...");
                    match get_recipe_from_anthropic(recipe_name).await {
                        Ok(recipe) => {
                            println!("Recipe successfully generated with Anthropic! 🎉");
                            print_recipe(&recipe);
                        }
                        Err(e) => println!("Error getting recipe from Anthropic: {}", e),
                    }
                } else {
                    println!("No ANTHROPIC_API_KEY set for fallback.");
                }
            }
        }
    } else if env::var("ANTHROPIC_API_KEY").is_ok() {
        match get_recipe_from_anthropic(recipe_name).await {
            Ok(recipe) => {
                println!("Recipe successfully generated with Anthropic! 🎉");
                print_recipe(&recipe);
            }
            Err(e) => println!("Error getting recipe from Anthropic: {}", e),
        }
    } else {
        println!(
            "Error: No API keys found. Please set either OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable."
        );
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    print!("What would you like a recipe for? ");
    io::stdout().flush()?;

    let mut recipe_name = String::new();
    io::stdin().read_line(&mut recipe_name)?;
    let recipe_name = recipe_name.trim();

    if recipe_name.is_empty() {
        println!("No recipe name entered. Using default: chocolate chip cookies");
        let recipe_name = "chocolate chip cookies".to_string();
        return run_recipe_extraction(&recipe_name).await;
    }

    run_recipe_extraction(recipe_name).await
}
