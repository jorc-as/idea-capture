use std::path::PathBuf;

use chrono::{DateTime, Utc};
use rust_bert::pegasus::{
    PegasusConditionalGenerator, PegasusConfigResources, PegasusModelResources,
    PegasusVocabResources,
};
use rust_bert::pipelines::generation_utils::{GenerateConfig, GenerateOptions, LanguageGenerator};
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsConfig, SentenceEmbeddingsModelType,
};
use rust_bert::resources::{LocalResource, RemoteResource};
use tch::Device;
// Enhanced Note structure with more metadata
#[derive(Debug, Clone)]
pub struct Note {
    text: String,
    created_at: DateTime<Utc>,
    title: Option<String>,
    tags: Vec<String>,
    category: NoteCategory,
    speaker: Option<String>,
    importance: Importance,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NoteCategory {
    ActionItem,
    Information,
    Decision,
    Question,
    Other(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Importance {
    High,
    Medium,
    Low,
}

// User preferences for personalization
#[derive(Debug, Clone)]
pub struct UserPreferences {
    voice_style: VoiceStyle,
    summarization_length: SummarizationLength,
    highlight_action_items: bool,
    organize_by_topic: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VoiceStyle {
    FirstPerson,
    ThirdPerson,
    Neutral,
}

#[derive(Debug, Clone)]
pub enum SummarizationLength {
    Brief,    // ~25% original
    Moderate, // ~50% original
    Detailed, // ~75% original
}

// Main note processor
pub struct NoteTaker {
    model: PegasusConditionalGenerator,
    user_prefs: UserPreferences,
    sentence_embedder:
        Option<Box<rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel>>,
}

impl NoteTaker {
    pub fn new(user_prefs: UserPreferences) -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize model
        let model_rsrc = Box::new(LocalResource {
            local_path: PathBuf::from("../../../ml_models/rust_model.ot"),
        });
        let config_rsrc = Box::new(LocalResource {
            local_path: PathBuf::from("../../../ml_models/config.json"),
        });
        let vocab_rsrc = Box::new(LocalResource {
            local_path: PathBuf::from("../../../ml_models/spiece.model"),
        });

        let device = Device::cuda_if_available();

        let generate_config = GenerateConfig {
            model_resource: rust_bert::pipelines::common::ModelResource::Torch(model_rsrc),
            config_resource: config_rsrc,
            vocab_resource: vocab_rsrc,
            max_length: Some(100),
            num_beams: 4, // Increase beam search for better results
            no_repeat_ngram_size: 3,
            device,
            ..Default::default()
        };
        println!("1");
        let model = PegasusConditionalGenerator::new(generate_config)?;
        println!("2");
        // Initialize sentence embeddings model
        let sentence_embedder =
            SentenceEmbeddingsBuilder::local("../../../ml_models/sentenceMini.ot")
                .with_device(Device::cuda_if_available())
                .create_model()
                .ok()
                .map(Box::new);
        Ok(NoteTaker {
            model,
            user_prefs,
            sentence_embedder,
        })
    }

    pub fn process_transcript(
        &self,
        transcript: &str,
    ) -> Result<Vec<Note>, Box<dyn std::error::Error>> {
        // Segment the transcript
        let segments = self.segment_transcript(transcript);

        let mut notes = Vec::new();

        for segment in segments {
            // Classify segment
            let category = self.classify_segment(&segment);

            // Determine if summarization is needed based on length
            let processed_text = if segment.split_whitespace().count() > 30 {
                self.summarize(&segment)?
            } else {
                segment.clone()
            };

            // Apply personalization
            let personalized_text = self.personalize_text(&processed_text);

            // Extract a title
            let title = self.extract_title(&processed_text);

            // Create note
            let note = Note {
                text: personalized_text,
                created_at: Utc::now(),
                title: Some(title),
                tags: self.extract_tags(&processed_text),
                category,
                speaker: None, // Could be determined from audio with speaker diarization
                importance: self.determine_importance(&segment),
            };

            notes.push(note);
        }

        // Organize notes if needed
        if self.user_prefs.organize_by_topic {
            self.organize_by_topic(&mut notes);
        }

        Ok(notes)
    }
    pub fn segment_transcript(&self, transcript: &str) -> Vec<String> {
        // Step 1: Split transcript into initial sentences
        println!("Segmenting the transcript");
        let raw_sentences = transcript
            .replace("? ", "?|")
            .replace("! ", "!|")
            .replace(". ", ".|");
        let bind: Vec<&str> = raw_sentences
            .split('|')
            .filter(|s| !s.trim().is_empty())
            .collect();

        let sentences: Vec<String> = bind.iter().map(|&s| s.trim().to_string()).collect();

        // If we have embedder, use semantic segmentation
        if let Some(embedder) = &self.sentence_embedder {
            return self
                .semantic_segmentation(embedder, &sentences)
                .unwrap_or_else(|_| self.fallback_segmentation(&sentences));
        }

        // Fallback to simpler segmentation if no embedder
        self.fallback_segmentation(&sentences)
    }

    fn semantic_segmentation(
        &self,
        embedder: &Box<rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel>,
        sentences: &[String],
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        println!("semantic");
        // Generate embeddings for all sentences
        let embeddings = embedder.encode(sentences)?;

        // Convert to Vec<Vec<f64>> for easier handling
        let embeddings: Vec<Vec<f64>> = embeddings
            .iter()
            .map(|v| v.iter().map(|&f| f as f64).collect())
            .collect();

        const SIMILARITY_THRESHOLD: f64 = 0.5;
        const MAX_SEGMENT_SIZE: usize = 5; // Maximum number of sentences per segment

        let mut segments = Vec::new();
        let mut current_segment = Vec::new();
        let mut segment_embedding = Vec::new();

        // Initialize with first sentence
        if !sentences.is_empty() {
            current_segment.push(sentences[0].clone());
            segment_embedding = embeddings[0].clone();
        }

        // Process remaining sentences
        for i in 1..sentences.len() {
            // Calculate similarity with current segment
            let similarity = cosine_similarity(&segment_embedding, &embeddings[i]);

            if similarity >= SIMILARITY_THRESHOLD && current_segment.len() < MAX_SEGMENT_SIZE {
                // Add to current segment if similar enough
                current_segment.push(sentences[i].clone());

                // Update segment embedding (average of all sentences in segment)
                segment_embedding = average_embeddings(
                    &current_segment
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, _)| {
                            // Map back to original position in sentences
                            let orig_idx = if idx == current_segment.len() - 1 {
                                i // Just added sentence
                            } else {
                                idx + i - current_segment.len() + 1 // Previously added sentences
                            };

                            if orig_idx < embeddings.len() {
                                Some(embeddings[orig_idx].clone())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<Vec<f64>>>(),
                );
            } else {
                // Start a new segment
                if !current_segment.is_empty() {
                    segments.push(current_segment.join(" "));
                    current_segment = Vec::new();
                }

                current_segment.push(sentences[i].clone());
                segment_embedding = embeddings[i].clone();
            }
        }

        // Add the final segment
        if !current_segment.is_empty() {
            segments.push(current_segment.join(" "));
        }

        Ok(segments)
    }

    fn fallback_segmentation(&self, sentences: &[String]) -> Vec<String> {
        // A fallback method that doesn't require embeddings
        // Group sentences using topic shift markers
        let topic_markers = [
            "next",
            "also",
            "additionally",
            "moving on",
            "furthermore",
            "oh",
            "right",
            "by the way",
            "another thing",
            "anyway",
        ];

        let mut segments = Vec::new();
        let mut current_segment = Vec::new();

        for sentence in sentences {
            println!("segmenting");
            let lower = sentence.to_lowercase();
            let is_topic_shift = topic_markers
                .iter()
                .any(|&marker| lower.starts_with(marker));

            if is_topic_shift && !current_segment.is_empty() {
                segments.push(current_segment.join(" "));
                current_segment = Vec::new();
            }

            current_segment.push(sentence.clone());
        }

        // Add the final segment
        if !current_segment.is_empty() {
            segments.push(current_segment.join(" "));
        }

        segments
    }

    fn summarize(&self, text: &str) -> Result<String, Box<dyn std::error::Error>> {
        // Adjust max length based on user preference
        let max_length = match self.user_prefs.summarization_length {
            SummarizationLength::Brief => 30,
            SummarizationLength::Moderate => 50,
            SummarizationLength::Detailed => 80,
        };

        let options = GenerateOptions {
            max_length: Some(max_length),
            num_beams: Some(4),
            num_return_sequences: Some(1),
            ..Default::default()
        };

        let input = &[text];
        let output = self.model.generate(Some(input), Some(options))?;

        if output.is_empty() {
            return Err("Failed to generate summary".into());
        }

        Ok(output[0].text.clone())
    }

    fn personalize_text(&self, text: &str) -> String {
        match self.user_prefs.voice_style {
            VoiceStyle::FirstPerson => {
                text.replace("The speaker said", "I said")
                    .replace("They mentioned", "I mentioned")
                    .replace("The participant noted", "I noted")
                // Add more replacements as needed
            }
            VoiceStyle::ThirdPerson => text.to_string(), // Already in third person typically
            VoiceStyle::Neutral => {
                // Remove personal references
                text.replace("I think", "It was mentioned")
                    .replace("I believe", "It was stated")
                // Add more neutralizing replacements
            }
        }
    }

    fn classify_segment(&self, text: &str) -> NoteCategory {
        let lower_text = text.to_lowercase();

        // Simple keyword-based classification
        if lower_text.contains("action")
            || lower_text.contains("todo")
            || lower_text.contains("task")
        {
            NoteCategory::ActionItem
        } else if lower_text.contains("decid")
            || lower_text.contains("conclusion")
            || lower_text.contains("resolve")
        {
            NoteCategory::Decision
        } else if lower_text.contains("question")
            || lower_text.contains("?")
            || lower_text.contains("wonder")
        {
            NoteCategory::Question
        } else {
            NoteCategory::Information
        }
    }

    fn extract_title(&self, text: &str) -> String {
        // Simple implementation: use first sentence or first few words
        let first_sentence = text
            .split(|c| c == '.' || c == '?' || c == '!')
            .next()
            .unwrap_or(text);

        if first_sentence.len() > 30 {
            format!("{}...", &first_sentence[0..30])
        } else {
            first_sentence.to_string()
        }
    }

    fn extract_tags(&self, text: &str) -> Vec<String> {
        let lower_text = text.to_lowercase();
        let mut tags = Vec::new();

        // Simple keyword extraction
        // In a real implementation, you'd use NLP techniques
        let common_topics = [
            "meeting",
            "project",
            "deadline",
            "budget",
            "client",
            "feature",
            "bug",
            "report",
            "presentation",
        ];

        for topic in common_topics.iter() {
            if lower_text.contains(topic) {
                tags.push(topic.to_string());
            }
        }

        tags
    }

    fn determine_importance(&self, text: &str) -> Importance {
        let lower_text = text.to_lowercase();

        // Simple keyword-based importance determination
        if lower_text.contains("important")
            || lower_text.contains("critical")
            || lower_text.contains("urgent")
        {
            Importance::High
        } else if lower_text.contains("should")
            || lower_text.contains("need to")
            || lower_text.contains("remember")
        {
            Importance::Medium
        } else {
            Importance::Low
        }
    }

    fn organize_by_topic(&self, notes: &mut Vec<Note>) {
        // Group related notes based on tags or content similarity
        // This is a simple implementation - in reality you might use
        // more sophisticated clustering algorithms

        notes.sort_by(|a, b| {
            // First sort by category
            let category_order = |cat: &NoteCategory| -> u8 {
                match cat {
                    NoteCategory::ActionItem => 0,
                    NoteCategory::Decision => 1,
                    NoteCategory::Question => 2,
                    NoteCategory::Information => 3,
                    NoteCategory::Other(_) => 4,
                }
            };

            let cat_cmp = category_order(&a.category).cmp(&category_order(&b.category));
            if cat_cmp != std::cmp::Ordering::Equal {
                return cat_cmp;
            }

            // Then by importance
            let importance_order = |imp: &Importance| -> u8 {
                match imp {
                    Importance::High => 0,
                    Importance::Medium => 1,
                    Importance::Low => 2,
                }
            };

            importance_order(&a.importance).cmp(&importance_order(&b.importance))
        });
    }
}
fn cosine_similarity(v1: &[f64], v2: &[f64]) -> f64 {
    if v1.is_empty() || v2.is_empty() {
        return 0.0;
    }

    let mut dot_product = 0.0;
    let mut norm_v1 = 0.0;
    let mut norm_v2 = 0.0;

    let min_len = v1.len().min(v2.len());

    for i in 0..min_len {
        dot_product += v1[i] * v2[i];
        norm_v1 += v1[i] * v1[i];
        norm_v2 += v2[i] * v2[i];
    }

    let norm_v1 = norm_v1.sqrt();
    let norm_v2 = norm_v2.sqrt();

    if norm_v1 > 0.0 && norm_v2 > 0.0 {
        dot_product / (norm_v1 * norm_v2)
    } else {
        0.0
    }
}

fn average_embeddings(embeddings: &[Vec<f64>]) -> Vec<f64> {
    if embeddings.is_empty() {
        return Vec::new();
    }

    let dim = embeddings[0].len();
    let mut avg = vec![0.0; dim];

    for embedding in embeddings {
        for i in 0..dim {
            if i < embedding.len() {
                avg[i] += embedding[i];
            }
        }
    }

    for i in 0..dim {
        avg[i] /= embeddings.len() as f64;
    }

    avg
}

// Main function to process audio transcripts
pub fn process_audio_transcript(transcript: &str) -> Result<Vec<Note>, Box<dyn std::error::Error>> {
    // Create default user preferences
    let user_prefs = UserPreferences {
        voice_style: VoiceStyle::FirstPerson,
        summarization_length: SummarizationLength::Moderate,
        highlight_action_items: true,
        organize_by_topic: true,
    };

    // Initialize note taker
    let note_taker = NoteTaker::new(user_prefs)?;

    // Process transcript
    note_taker.process_transcript(transcript)
}

// Function to format notes for display or export
pub fn format_notes(notes: &[Note], format: &str) -> String {
    match format {
        "markdown" => {
            let mut output = String::new();
            output.push_str("# Meeting Notes\n\n");

            // Group notes by category
            let mut action_items = Vec::new();
            let mut decisions = Vec::new();
            let mut questions = Vec::new();
            let mut information = Vec::new();

            for note in notes {
                match note.category {
                    NoteCategory::ActionItem => action_items.push(note),
                    NoteCategory::Decision => decisions.push(note),
                    NoteCategory::Question => questions.push(note),
                    NoteCategory::Information => information.push(note),
                    _ => information.push(note),
                }
            }

            // Output action items
            if !action_items.is_empty() {
                output.push_str("## Action Items\n\n");
                for note in action_items {
                    output.push_str(&format!(
                        "- **{}**: {}\n",
                        note.title.as_ref().unwrap_or(&"Action".to_string()),
                        note.text
                    ));
                }
                output.push('\n');
            }

            // Output decisions
            if !decisions.is_empty() {
                output.push_str("## Decisions\n\n");
                for note in decisions {
                    output.push_str(&format!(
                        "- **{}**: {}\n",
                        note.title.as_ref().unwrap_or(&"Decision".to_string()),
                        note.text
                    ));
                }
                output.push('\n');
            }

            // Output questions
            if !questions.is_empty() {
                output.push_str("## Questions\n\n");
                for note in questions {
                    output.push_str(&format!(
                        "- **{}**: {}\n",
                        note.title.as_ref().unwrap_or(&"Question".to_string()),
                        note.text
                    ));
                }
                output.push('\n');
            }

            // Output information
            if !information.is_empty() {
                output.push_str("## Key Information\n\n");
                for note in information {
                    output.push_str(&format!(
                        "- **{}**: {}\n",
                        note.title.as_ref().unwrap_or(&"Note".to_string()),
                        note.text
                    ));
                }
            }

            output
        }
        "json" => {
            // In a real implementation, use a proper JSON serialization library
            let mut output = String::from("[\n");

            for (i, note) in notes.iter().enumerate() {
                output.push_str(&format!(
                    "  {{\n    \"title\": \"{}\",\n    \"text\": \"{}\",\n    \"category\": \"{:?}\"\n  }}",
                    note.title.as_ref().unwrap_or(&"".to_string()).replace("\"", "\\\""),
                    note.text.replace("\"", "\\\""),
                    note.category
                ));

                if i < notes.len() - 1 {
                    output.push_str(",\n");
                } else {
                    output.push_str("\n");
                }
            }

            output.push_str("]\n");
            output
        }
        _ => {
            // Plain text format
            let mut output = String::new();
            output.push_str("MEETING NOTES\n\n");

            for note in notes {
                let category_str = match note.category {
                    NoteCategory::ActionItem => "[ACTION]",
                    NoteCategory::Decision => "[DECISION]",
                    NoteCategory::Question => "[QUESTION]",
                    NoteCategory::Information => "[INFO]",
                    NoteCategory::Other(ref s) => s,
                };

                output.push_str(&format!(
                    "{} {}: {}\n\n",
                    category_str,
                    note.title.as_ref().unwrap_or(&"".to_string()),
                    note.text
                ));
            }

            output
        }
    }
}

// Example usage function
pub fn process_transcript_example() -> Result<(), Box<dyn std::error::Error>> {
    let transcript = "First, I really need to email Sarah about that project timeline. Just ask if we’re still aiming for next Friday, or if that’s getting pushed. Don’t assume—just double-check, 'cause if it moves and I don’t know, I’m gonna look like an idiot. Also, maybe ask if she got the updated deck I sent last week.

Next—groceries. Definitely stop by the store on the way home. We’re out of coffee—again. Gotta get the dark roast, not that light stuff. Also out of milk... and probably eggs. Actually, just do a quick check of the fridge before leaving work, so I don’t buy stuff we already have.

Oh, and I need to remember to pick up the dry cleaning. I think it’s been like... a week? If I wait any longer they’ll think I forgot completely. That navy jacket should be ready.

What else... right, the package! Check the mailroom first, if it’s not there, check the front desk. If nobody signed for it, look up the tracking number and maybe give the courier a call—ugh. Hopefully it’s just there.

Also, random reminder—back up the files from the external hard drive tonight. I’ve been putting that off forever. If I lose that stuff, I’m toast.

Oh! Call Mom. Just to check in. Don’t let it go another few days, she’ll guilt trip me into oblivion.

Alright, I think that’s it. Probably missing something, but I’ll add more if it comes to me. Don’t ignore this. Seriously. Do the things.";

    let notes = process_audio_transcript(transcript)?;

    println!("Generated {} notes", notes.len());

    // Format as markdown
    for note in notes {
        println!("{:?}", note.importance);

        println!("{}", note.text);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_note_creation() {
        let note = Note {
            text: "Test note".to_string(),
            created_at: Utc::now(),
            title: Some("Test".to_string()),
            tags: vec!["test".to_string()],
            category: NoteCategory::Information,
            speaker: None,
            importance: Importance::Medium,
        };
        assert!(process_transcript_example().is_ok());
        assert_eq!(note.text, "Test note");
        assert_eq!(note.category, NoteCategory::Information);
    }

    // More comprehensive tests would test the various components
    // but many depend on the Pegasus model which makes testing complex
}
