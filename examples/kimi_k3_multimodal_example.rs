//! Read a chart with Moonshot Kimi K3 and materialize typed Rust data.
//!
//! Run with:
//!
//! ```text
//! MOONSHOT_API_KEY=your_key_here \
//!   cargo run --example kimi_k3_multimodal_example --features openai
//! ```
//!
//! The example exits without a request when `MOONSHOT_API_KEY` is unset, which
//! keeps CI deterministic. Kimi K3 accepts inline base64 images but not public
//! image URLs, so the chart is downloaded before it is attached. Moonshot
//! documents JPEG, PNG, GIF, WebP, BMP, HEIC, and HEIF support (not SVG), and
//! recommends keeping images at or below 4096×2160.

use rstructor::{Instructor, LLMClient, MediaFile, OpenAIClient};
use serde::{Deserialize, Serialize};

const CHART_CONFIG: &str = r#"{
  type: 'bar',
  data: {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    datasets: [{
      label: 'Revenue (USD millions)',
      data: [2.4, 3.1, 2.8, 4.2, 4.9, 5.6],
      backgroundColor: ['#38bdf8', '#38bdf8', '#f97316', '#38bdf8', '#38bdf8', '#22c55e'],
      borderColor: '#0f172a',
      borderWidth: 1
    }]
  },
  options: {
    plugins: {
      title: { display: true, text: '2026 Revenue by Month', font: { size: 22 } },
      legend: { display: true },
      datalabels: {
        anchor: 'end',
        align: 'top',
        formatter: (value) => '$' + value + 'M',
        font: { weight: 'bold' }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        suggestedMax: 6.3,
        title: { display: true, text: 'USD millions' }
      },
      x: { title: { display: true, text: 'Month' } }
    }
  }
}"#;

#[derive(Debug, Deserialize, Instructor, Serialize)]
#[serde(rename_all = "snake_case")]
enum RevenueTrend {
    Rising,
    Falling,
    Flat,
    Mixed,
}

#[derive(Debug, Deserialize, Instructor, Serialize)]
struct MonthlyRevenue {
    #[llm(description = "Three-letter month label shown under the bar")]
    month: String,
    #[llm(description = "Revenue in USD millions, exactly as labeled on the chart")]
    revenue_millions: f64,
}

#[derive(Debug, Deserialize, Instructor, Serialize)]
#[llm(description = "Structured data and conclusions read from a revenue chart")]
struct RevenueChart {
    #[llm(description = "The chart title")]
    title: String,
    #[llm(description = "Every monthly bar, in left-to-right chart order")]
    monthly_revenue: Vec<MonthlyRevenue>,
    #[llm(description = "The month with the highest revenue")]
    peak_month: String,
    #[llm(description = "The highest revenue value in USD millions")]
    peak_revenue_millions: f64,
    #[llm(description = "The sum of all monthly revenue values in USD millions")]
    total_revenue_millions: f64,
    #[llm(description = "The overall direction across the full period")]
    overall_trend: RevenueTrend,
    #[llm(description = "One sentence identifying the most notable month-to-month change")]
    notable_change: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    match std::env::var("MOONSHOT_API_KEY") {
        Ok(key) if !key.trim().is_empty() => {}
        _ => {
            eprintln!("Skipping Kimi K3 request. Set MOONSHOT_API_KEY to run this example.");
            return Ok(());
        }
    }

    let chart = reqwest::Client::new()
        .get("https://quickchart.io/chart")
        .query(&[
            ("c", CHART_CONFIG),
            ("version", "4"),
            ("width", "900"),
            ("height", "500"),
            ("backgroundColor", "white"),
            ("format", "png"),
        ])
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;
    let media = MediaFile::from_bytes(&chart, "image/png");

    // Kimi K3 fixes temperature at 1.0. Setting it explicitly keeps the
    // OpenAI-compatible request within Moonshot's documented parameter limits.
    let client = OpenAIClient::moonshot()?.model("kimi-k3").temperature(1.0);

    let report: RevenueChart = client
        .materialize_with_media(
            "Read every labeled bar exactly. Calculate total revenue, identify \
             the peak month, classify the overall trend, and call out the most \
             notable month-to-month change.",
            &[media],
        )
        .await?;

    println!("{report:#?}");
    Ok(())
}
