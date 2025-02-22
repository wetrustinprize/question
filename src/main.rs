use clap::{builder::StyledStr, Parser};
use console::{style, Term};
use ollama_rs::{
    generation::completion::{request::GenerationRequest, GenerationResponse},
    IntoUrlSealed, Ollama,
};
use std::{
    io::{stdout, Write},
    thread::ThreadId,
};
use tokio::io::{self, AsyncWriteExt};
use tokio_stream::StreamExt;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(value_parser = check_prompt)]
    prompt: String,
}

fn check_prompt(prompt: &str) -> Result<String, String> {
    if prompt.is_empty() {
        return Err("Prompt is empty".to_string());
    }
    Ok(prompt.to_string())
}

const THINK_PROMPT: &str = "🤔 ";
const RESPONSE_PROMPT: &str = "💬 ";

const SYSTEM: &str = "
	You are suposed to give quick responses unless the user asks for a longer response.
	Don't use emojis.
	Don't use markdown, you are running on a terminal.
";

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let ollama = Ollama::default();

    let model = "deepseek-r1:7b".to_string();
    let prompt = args.prompt.to_string();

    let mut request = GenerationRequest::new(model, prompt);
    request.system = Some(SYSTEM.into());

    let mut stream = ollama.generate_stream(request).await.unwrap();

    let mut term = Term::stdout();

    let mut thinking = false;
    let mut response = String::new();
    let mut response_started = false;

    while let Some(res) = stream.next().await {
        let responses = res.unwrap();
        for resp in responses {
            response.push_str(&resp.response);

            if response.ends_with("<think>") {
                thinking = true;

                response.clear();
            } else if response.ends_with("</think>") {
                thinking = false;
                term.clear_last_lines(1).unwrap();

                term.write(RESPONSE_PROMPT.as_bytes()).unwrap();

                response.clear();
            } else if response.ends_with("\n") {
                if thinking {
                    term.clear_line().unwrap();
                } else if response_started || !resp.response.starts_with('\n') {
                    term.write(resp.response.as_bytes()).unwrap();
                    response_started = true;
                }

                response.clear();
            } else {
                if thinking {
                    term.clear_last_lines(1).unwrap();

                    let columns = term.size().1;
                    let available_space = columns as usize - THINK_PROMPT.len();

                    let trimmed_response = if response.len() > available_space {
                        let chars: Vec<char> = response.chars().collect();
                        let start_index = chars.len().saturating_sub(available_space - 3);
                        format!("...{}", chars[start_index..].iter().collect::<String>())
                    } else {
                        response.clone()
                    };

                    term.write_line(
                        format!("{}{}", THINK_PROMPT, style(trimmed_response).dim().italic())
                            .as_str(),
                    )
                    .unwrap();
                } else {
                    term.write(resp.response.as_bytes()).unwrap();
                }
            }
        }
    }
}
