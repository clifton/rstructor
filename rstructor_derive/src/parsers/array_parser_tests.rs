#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;
    use syn::parse::Parser;
    use syn::parse_str;
    
    #[test]
    fn test_parse_string_array() {
        // Create a mock parser buffer from string
        let input = parse_str::<syn::ExprArray>("[\"apple\", \"banana\", \"cherry\"]").unwrap();
        
        // Convert to a token stream
        let tokens = quote! { #input };
        
        // Convert back to a parse buffer
        let buffer = syn::parse2(tokens).unwrap();
        
        // Parse array literal
        let result = parse_array_literal(&buffer);
        
        // Check result
        assert!(result.is_some());
        let tokens = result.unwrap();
        assert_eq!(tokens.len(), 3);
    }
    
    #[test]
    fn test_parse_int_array() {
        // Create a mock parser buffer from string
        let input = parse_str::<syn::ExprArray>("[1, 2, 3, 4]").unwrap();
        
        // Convert to a token stream
        let tokens = quote! { #input };
        
        // Convert back to a parse buffer
        let buffer = syn::parse2(tokens).unwrap();
        
        // Parse array literal
        let result = parse_array_literal(&buffer);
        
        // Check result
        assert!(result.is_some());
        let tokens = result.unwrap();
        assert_eq!(tokens.len(), 4);
    }
    
    #[test]
    fn test_parse_mixed_array() {
        // Create a mock parser buffer from string
        let input = parse_str::<syn::ExprArray>("[\"string\", 42, true]").unwrap();
        
        // Convert to a token stream
        let tokens = quote! { #input };
        
        // Convert back to a parse buffer
        let buffer = syn::parse2(tokens).unwrap();
        
        // Parse array literal
        let result = parse_array_literal(&buffer);
        
        // Check result
        assert!(result.is_some());
        let tokens = result.unwrap();
        assert_eq!(tokens.len(), 3);
    }
    
    #[test]
    fn test_parse_empty_array() {
        // Create a mock parser buffer from string
        let input = parse_str::<syn::ExprArray>("[]").unwrap();
        
        // Convert to a token stream
        let tokens = quote! { #input };
        
        // Convert back to a parse buffer
        let buffer = syn::parse2(tokens).unwrap();
        
        // Parse array literal
        let result = parse_array_literal(&buffer);
        
        // Check result
        assert!(result.is_some());
        let tokens = result.unwrap();
        assert_eq!(tokens.len(), 0);
    }
}