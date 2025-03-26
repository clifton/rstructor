use rstructor::{LLMModel, LLMClient, OpenAIClient, OpenAIModel, AnthropicClient, AnthropicModel};
use serde::{Serialize, Deserialize};
use std::env;

// Define an enum for article categories
#[derive(LLMModel, Serialize, Deserialize, Debug)]
#[llm(description = "Category for a news article",
      examples = ["Politics", "Technology", "Business", "Sports", "Entertainment"])]
enum ArticleCategory {
    Politics,
    Technology,
    Business,
    Sports,
    Entertainment,
    Health,
    Science,
    Environment,
    Education,
    Opinion,
    Other,
}

// Define entities mentioned in the article
#[derive(LLMModel, Serialize, Deserialize, Debug)]
#[llm(description = "An entity mentioned in the article")]
struct Entity {
    #[llm(description = "Name of the entity",
          example = "Microsoft")]
    name: String,
    
    #[llm(description = "Type of the entity (person, organization, location, etc.)",
          example = "organization")]
    entity_type: String,
    
    #[llm(description = "How important this entity is to the article (1-10 scale)",
          example = 8)]
    relevance: u8,
}

// Custom validation for Entity
impl Entity {
    fn validate(&self) -> rstructor::Result<()> {
        // Check that relevance is within the expected range (1-10)
        if self.relevance < 1 || self.relevance > 10 {
            return Err(rstructor::RStructorError::ValidationError(
                format!("Relevance must be between 1 and 10, got {}", self.relevance)
            ));
        }
        Ok(())
    }
}

// Define the structure for article analysis
#[derive(LLMModel, Serialize, Deserialize, Debug)]
#[llm(description = "Analysis of a news article",
      examples = [
        ::serde_json::json!({
            "title": "Apple Unveils New iPhone 15 with Revolutionary Camera Technology",
            "category": "Technology",
            "summary": "Apple announced its new iPhone 15 lineup featuring a groundbreaking camera system with a periscope lens for improved zoom capabilities.",
            "sentiment": "Positive",
            "entities": [
                {"name": "Apple", "entity_type": "organization", "relevance": 10},
                {"name": "iPhone 15", "entity_type": "product", "relevance": 9}
            ],
            "keywords": ["Apple", "iPhone", "camera", "technology", "smartphone"],
            "bias_assessment": "The article presents the product in a favorable light with minimal criticism of potential drawbacks or cost concerns."
        })
      ])]
struct ArticleAnalysis {
    #[llm(description = "Title of the article",
          example = "Tech Stocks Tumble as Inflation Fears Rise")]
    title: String,
    
    #[llm(description = "Category the article belongs to")]
    category: ArticleCategory,
    
    #[llm(description = "A brief summary of the article (2-3 sentences)",
          example = "The article discusses recent market movements in the technology sector. Major tech stocks fell by an average of 3% following concerns about rising inflation.")]
    summary: String,
    
    #[llm(description = "Overall sentiment of the article (Positive, Negative, or Neutral)",
          example = "Negative")]
    sentiment: String,
    
    #[llm(description = "Main entities mentioned in the article")]
    entities: Vec<Entity>,
    
    #[llm(description = "Important keywords from the article",
          example = ["stocks", "technology", "inflation", "market"])]
    keywords: Vec<String>,
    
    #[llm(description = "Assessment of any bias in the reporting",
          example = "The article presents a somewhat negative view of tech companies, with limited perspective from industry insiders.")]
    bias_assessment: String,
}

// Function to analyze an article using an LLM
async fn analyze_article(article_text: &str) -> Result<ArticleAnalysis, Box<dyn std::error::Error>> {
    // Try using available API keys
    if let Ok(api_key) = env::var("OPENAI_API_KEY") {
        println!("Using OpenAI for article analysis...");
        
        let client = OpenAIClient::new(api_key)?
            .model(OpenAIModel::Gpt4)
            .temperature(0.0)
            .build();
        
        let prompt = format!("Analyze the following news article:\n\n{}", article_text);
        Ok(client.generate_struct::<ArticleAnalysis>(&prompt).await?)
        
    } else if let Ok(api_key) = env::var("ANTHROPIC_API_KEY") {
        println!("Using Anthropic for article analysis...");
        
        let client = AnthropicClient::new(api_key)?
            .model(AnthropicModel::Claude3Sonnet)
            .temperature(0.0)
            .build();
        
        let prompt = format!("Analyze the following news article:\n\n{}", article_text);
        Ok(client.generate_struct::<ArticleAnalysis>(&prompt).await?)
        
    } else {
        Err("No API keys found. Please set either OPENAI_API_KEY or ANTHROPIC_API_KEY.".into())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Sample article text
    let article = r#"
    TECH GIANT UNVEILS REVOLUTIONARY AI CHIP AMID COMPETITION CONCERNS

    Silicon Valley, CA - Tech behemoth NeuraTech announced yesterday the release of their new quantum-
    based AI processor, the N-2000, which they claim can perform machine learning tasks at speeds 50 
    times faster than current market leaders while using 75% less energy.

    CEO Jane Rodriguez showcased the processor at their annual developers conference, demonstrating 
    its capabilities by training a large language model in minutes rather than days. "This represents 
    a fundamental shift in what's possible with artificial intelligence," Rodriguez told the crowd of 
    developers and investors.

    The announcement comes as regulatory bodies in both the US and EU are scrutinizing the growing 
    concentration of AI capabilities among a small number of tech companies. Last month, the Federal 
    Trade Commission opened an inquiry into potential anticompetitive practices in the AI chip market.

    Market analysts reacted positively to the news, with NeuraTech's stock price jumping 12% by closing 
    bell. "The efficiency gains here can't be overstated," said Morgan Stanley analyst Raj Patel. "If 
    the performance metrics hold up in real-world applications, this could reshape the competitive 
    landscape."

    Competing chip manufacturers SynthLogic and Quantum Semiconductors saw stock declines of 5% and 7% 
    respectively following the announcement. Representatives from both companies declined to comment.

    The N-2000 processor is expected to begin shipping to select enterprise customers in Q3, with wider 
    availability planned for early next year.
    "#;
    
    // Analyze the article
    match analyze_article(article).await {
        Ok(analysis) => {
            println!("\n===== Article Analysis =====");
            println!("Title: {}", analysis.title);
            println!("Category: {:?}", analysis.category);
            println!("\nSummary: {}", analysis.summary);
            println!("\nSentiment: {}", analysis.sentiment);
            
            println!("\nEntities:");
            for entity in analysis.entities {
                println!("• {} ({}): Relevance {}/10", 
                         entity.name, entity.entity_type, entity.relevance);
            }
            
            println!("\nKeywords: {}", analysis.keywords.join(", "));
            println!("\nBias Assessment: {}", analysis.bias_assessment);
        },
        Err(e) => {
            println!("Error analyzing article: {}", e);
        }
    }
    
    Ok(())
}