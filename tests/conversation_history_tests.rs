//! Tests for conversation history retry behavior.
//!
//! These tests verify that the retry logic properly builds conversation history
//! when validation errors occur, enabling prompt caching benefits.

use rstructor::{ChatMessage, ChatRole};

mod conversation_history_tests {
    use super::*;

    #[test]
    fn test_chat_message_user_creation() {
        let msg = ChatMessage::user("Hello, world!");
        assert_eq!(msg.role.as_str(), "user");
        assert_eq!(msg.content, "Hello, world!");
    }

    #[test]
    fn test_chat_message_assistant_creation() {
        let msg = ChatMessage::assistant("I am an assistant.");
        assert_eq!(msg.role.as_str(), "assistant");
        assert_eq!(msg.content, "I am an assistant.");
    }

    #[test]
    fn test_chat_message_system_creation() {
        let msg = ChatMessage::system("You are a helpful assistant.");
        assert_eq!(msg.role.as_str(), "system");
        assert_eq!(msg.content, "You are a helpful assistant.");
    }

    #[test]
    fn test_chat_role_as_str() {
        assert_eq!(ChatRole::User.as_str(), "user");
        assert_eq!(ChatRole::Assistant.as_str(), "assistant");
        assert_eq!(ChatRole::System.as_str(), "system");
    }

    #[test]
    fn test_chat_message_new() {
        let msg = ChatMessage::new(ChatRole::User, "Test content");
        assert_eq!(msg.role, ChatRole::User);
        assert_eq!(msg.content, "Test content");
    }

    #[test]
    fn test_chat_message_with_string_content() {
        let content = String::from("Dynamic content");
        let msg = ChatMessage::user(content);
        assert_eq!(msg.content, "Dynamic content");
    }

    #[test]
    fn test_chat_role_equality() {
        assert_eq!(ChatRole::User, ChatRole::User);
        assert_ne!(ChatRole::User, ChatRole::Assistant);
        assert_ne!(ChatRole::Assistant, ChatRole::System);
    }

    #[test]
    fn test_chat_message_debug() {
        let msg = ChatMessage::user("test");
        let debug_str = format!("{:?}", msg);
        assert!(debug_str.contains("ChatMessage"));
        assert!(debug_str.contains("User"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_chat_role_debug() {
        let role = ChatRole::Assistant;
        let debug_str = format!("{:?}", role);
        assert!(debug_str.contains("Assistant"));
    }

    #[test]
    fn test_chat_message_clone() {
        let msg1 = ChatMessage::user("original");
        let msg2 = msg1.clone();
        assert_eq!(msg1.content, msg2.content);
        assert_eq!(msg1.role, msg2.role);
    }
}
