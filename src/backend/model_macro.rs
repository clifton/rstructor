//! Internal helper for declaring provider model enums without repetition.
//!
//! Every backend (OpenAI, Anthropic, Grok, Gemini) exposes a `Model` enum that
//! maps a set of named variants to their exact API identifiers and back. Before
//! this macro, each backend hand-wrote the enum, `as_str`, `from_string`,
//! `FromStr`, `From<&str>` and `From<String>` — roughly 60 lines of identical
//! boilerplate per provider, four times over, each a place to drift out of sync.
//!
//! [`define_model_enum!`] generates all of that from a single declaration, so a
//! backend only states its variants and their identifiers.

/// Generate a provider `Model` enum plus its string conversions.
///
/// Produces the enum (with the given doc comments and a trailing
/// `Custom(String)` variant), and implementations of `as_str`, `from_string`,
/// [`FromStr`](std::str::FromStr), `From<&str>` and `From<String>`.
///
/// Each named variant maps to/from its exact API identifier; any unrecognized
/// string round-trips losslessly through `Custom`.
///
/// ```text
/// define_model_enum! {
///     /// Example models.
///     pub enum Model {
///         /// The flagship model.
///         Flagship => "vendor-flagship",
///         /// A faster, cheaper model.
///         Mini => "vendor-mini",
///     }
/// }
/// ```
macro_rules! define_model_enum {
    (
        $(#[$enum_meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident => $model_id:literal,
            )+
        }
    ) => {
        $(#[$enum_meta])*
        #[derive(Debug, Clone, PartialEq, Eq)]
        $vis enum $name {
            $(
                $(#[$variant_meta])*
                $variant,
            )+
            /// Custom model identifier — for new models, local LLMs, or
            /// provider-compatible endpoints not covered by a named variant.
            Custom(String),
        }

        impl $name {
            /// Return the exact API model identifier for this model.
            pub fn as_str(&self) -> &str {
                match self {
                    $( $name::$variant => $model_id, )+
                    $name::Custom(name) => name,
                }
            }

            /// Create a model from a string.
            ///
            /// This convenience constructor always succeeds: a known identifier
            /// maps to its variant, and anything else becomes
            /// [`Custom`](Self::Custom).
            pub fn from_string(name: impl Into<String>) -> Self {
                let name = name.into();
                match name.as_str() {
                    $( $model_id => $name::$variant, )+
                    _ => $name::Custom(name),
                }
            }
        }

        impl ::std::str::FromStr for $name {
            type Err = ::std::convert::Infallible;

            fn from_str(s: &str) -> ::std::result::Result<Self, Self::Err> {
                Ok($name::from_string(s))
            }
        }

        impl ::std::convert::From<&str> for $name {
            fn from(s: &str) -> Self {
                $name::from_string(s)
            }
        }

        impl ::std::convert::From<String> for $name {
            fn from(s: String) -> Self {
                $name::from_string(s)
            }
        }
    };
}

pub(crate) use define_model_enum;
