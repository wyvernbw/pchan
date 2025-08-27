extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, Expr, ExprLit, Ident, Lit, Variant, parse_macro_input};

enum ValidRepr {
    U8,
}

impl ValidRepr {
    const fn max(&self) -> usize {
        match self {
            ValidRepr::U8 => u8::MAX as usize,
            ValidRepr::U16 => u16::MAX as usize,
            ValidRepr::U32 => u32::MAX as usize,
        }
    }
}

#[proc_macro_derive(OpCode, attributes(opcode))]
pub fn derive_opcode(stream: TokenStream) -> TokenStream {
    let stream = parse_macro_input!(stream as DeriveInput);
    let name = stream.ident.clone();
    let repr = stream.attrs.iter().find_map(|attr| {
        let mut repr = None;
        if attr.path().is_ident("repr") {
            let _ = attr.parse_nested_meta(|meta| {
                let Some(ident) = meta.path.get_ident() else {
                    return Ok(());
                };
                let ident = ident.to_string();
                use ValidRepr::*;
                let res = match ident.as_str() {
                    "u8" => Some(U8),
                    "u16" => Some(U16),
                    "u32" => Some(U32),
                    _ => None,
                };
                repr = res;
                Ok(())
            });
        }
        repr
    });

    let repr = repr.expect("OpCode derive macro requires an explicit repr attribute.");
    let syn::Data::Enum(data) = stream.data else {
        panic!("OpCode derive macro expects an enum.");
    };

    fn is_op_default(variant: &Variant) -> bool {
        variant
            .attrs
            .iter()
            .filter(|attr| {
                attr.path().is_ident("opcode")
                    && attr
                        .meta
                        .require_list()
                        .and_then(|meta| meta.parse_args::<Ident>())
                        .map(|ident| ident.to_string() == "default")
                        .unwrap_or(false)
            })
            .count()
            == 1
    }

    let default_value = data
        .variants
        .iter()
        .find(|variant| is_op_default(variant))
        .expect("OpCode macro requires one variant with the #[opcode(default)] attribute");
    let mut codes = vec![default_value.ident.clone(); repr.max()];
    for variant in data.variants.iter() {
        if is_op_default(variant) {
            continue;
        }
        let Some((_, discriminant)) = &variant.discriminant else {
            panic!("OpCode derive macro requires explicit discriminant values for enum variants.")
        };
        let value = match discriminant {
            Expr::Lit(ExprLit {
                lit: Lit::Int(value),
                ..
            }) => value
                .base10_parse::<usize>()
                .expect("failed to parse discriminant literal"),
            _ => panic!("expected discriminant value."),
        };
        codes[value] = variant.ident.clone();
    }

    let size = codes.len();
    let output = quote! {
        impl #name {
            const MAP: [#name; #size] = {
                use #name::*;
                [
                    #(#codes),*
                ]
            };

        }
    };

    TokenStream::from(output)
}

#[proc_macro_attribute]
pub fn opcode(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[test]
fn test_macro() {}
