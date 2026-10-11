extern crate alloc;

use alloc::borrow::Cow;

use darling::{FromDeriveInput, FromField, FromMeta, FromVariant, ast};
use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::DeriveInput;

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(endecode), supports(struct_named, struct_tuple))]
struct NamedStructOpts {
    ident:    syn::Ident,
    generics: syn::Generics,
    data:     ast::Data<(), EncodingField>,
}

#[derive(Debug, FromField, Clone)]
#[darling(attributes(endecode))]
struct EncodingField {
    ident: Option<syn::Ident>,
    skip:  Option<()>,
    ty:    syn::Type,
    alloc: Option<Alloc>,
}

#[derive(Debug, FromMeta, Clone)]
struct Alloc {
    with:   Option<syn::Ident>,
    bounds: Option<AllocBound>,
}

#[derive(Default, Debug, Clone, FromMeta)]
enum AllocBound {
    #[default]
    Clone,
    Copy,
}

fn min_bound<'a>(fields: impl IntoIterator<Item = &'a EncodingField>) -> AllocBound {
    let copy = fields.into_iter().any(|field| {
        field
            .alloc
            .as_ref()
            .is_some_and(|a| matches!(a.bounds, Some(AllocBound::Copy)))
    });
    if copy {
        AllocBound::Copy
    } else {
        AllocBound::Clone
    }
}

impl ToTokens for AllocBound {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let t = match self {
            AllocBound::Clone => quote! { Clone },
            AllocBound::Copy => quote! { Copy },
        };
        tokens.extend(t);
    }
}

struct EncodingUtilGeneric {
    typename:        syn::Ident,
    generics:        syn::Generics,
    where_clause:    Option<syn::WhereClause>,
    type_w_generics: proc_macro2::TokenStream,
    type_params:     Vec<proc_macro2::TokenStream>,
    allocator_bound: proc_macro2::TokenStream,
    endecode_bound:  proc_macro2::TokenStream,
}

impl EncodingUtilGeneric {
    fn from_ident_and_generics(ident: syn::Ident, generics: syn::Generics) -> Self {
        let typename = ident.clone();
        let generics = generics.clone();
        let (impl_generics, type_generics, where_c) = generics.split_for_impl();
        let where_clause = where_c.cloned();
        let type_params = generics
            .type_params()
            .filter_map(|ty| {
                let ident = &ty.ident;
                if &ident.to_string() == "A" {
                    return None;
                }
                let bounds = &ty.bounds;
                Some(quote! {
                    #ident : #bounds
                })
            })
            .collect::<Vec<_>>();
        let type_w_generics = quote! {
            #typename #type_generics
        };
        let allocator_bound = quote! {
            ::core::alloc::Allocator
        };
        let endecode_bound = quote! {
            ::pchan_encode::Endecode
        };
        EncodingUtilGeneric {
            typename,
            generics,
            where_clause,
            type_w_generics,
            type_params,
            allocator_bound,
            endecode_bound,
        }
    }
}

impl From<&NamedStructOpts> for EncodingUtilGeneric {
    fn from(value: &NamedStructOpts) -> Self {
        Self::from_ident_and_generics(value.ident.clone(), value.generics.clone())
    }
}

impl EncodingField {
    fn to_tokens(&self, index: Option<usize>) -> Option<proc_macro2::TokenStream> {
        let ident = &self.ident;
        if self.skip.is_some() {
            return None;
        }
        match ident {
            Some(ident) => Some(quote! {
                #ident.endecode(sink, alloc.clone())?;
            }),
            None => {
                let ident = syn::Index::from(index.expect("field with no name must be provided an index. this is a bug in the derive proc macro."));
                Some(quote! {
                    #ident.endecode(sink, alloc.clone())?;
                })
            }
        }
    }
}

impl ToTokens for NamedStructOpts {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let EncodingUtilGeneric {
            typename,
            generics,
            where_clause,
            type_w_generics,
            type_params,
            allocator_bound,
            endecode_bound,
        } = self.into();

        let fields = &self.data.as_struct().unwrap().fields;
        let fields_encode = fields
            .iter()
            .enumerate()
            .filter_map(|(i, field)| field.to_tokens(Some(i)));
        let fields_size = fields.iter().filter_map(|field| {
            let ty = &field.ty;
            if field.skip.is_some() {
                return None;
            }
            Some(quote! {
                <#ty as ::pchan_encode::Endecode<A>>::size()?
            })
        });
        let result = quote! {
            impl<A: #allocator_bound + Clone, #(#type_params)*> #endecode_bound<A> for #type_w_generics #where_clause {
                fn endecode(&mut self, sink: &mut impl ::pchan_encode::Stream, alloc: A) -> Result<(), ::pchan_encode::EndecodeError> {
                    #(self. #fields_encode)*
                    Ok(())
                }
                fn size() -> Result<Option<usize>, ::pchan_encode::EndecodeError> {
                    let mut size = 0;
                    #(
                    match #fields_size {
                        Some(s) => size += s,
                        None => return Ok(None)
                    };
                    )*
                    Ok(Some(size))
                }
            }
        };
        tokens.extend(result);
    }
}

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(endecode), supports(enum_any))]
struct EnumOpts {
    ident:    syn::Ident,
    generics: syn::Generics,
    data:     ast::Data<EncodingVariant, ()>,
}

#[derive(Debug, FromVariant)]
struct EncodingVariant {
    ident:  syn::Ident,
    skip:   Option<()>,
    fields: ast::Fields<EncodingField>,
}

trait FieldNames {
    fn field_names(&self) -> Vec<syn::Ident>;
}

impl FieldNames for ast::Fields<EncodingField> {
    fn field_names(&self) -> Vec<syn::Ident> {
        self.fields
            .iter()
            .enumerate()
            .map(|(i, field)| match &field.ident {
                Some(ident) => ident.clone(),
                None => {
                    let ident = format_ident!("_encode_{i}");
                    ident
                }
            })
            .collect::<Vec<_>>()
    }
}

impl EncodingVariant {
    fn destructure(&self, field_names: Option<&[syn::Ident]>) -> proc_macro2::TokenStream {
        let field_names = match field_names {
            Some(idents) => Cow::Borrowed(idents),
            None => Cow::Owned(self.fields.field_names()),
        };
        if self.fields.is_empty() {
            quote! {}
        } else {
            quote! { (#(#field_names),*) }
        }
    }
}

impl From<&EnumOpts> for EncodingUtilGeneric {
    fn from(value: &EnumOpts) -> Self {
        Self::from_ident_and_generics(value.ident.clone(), value.generics.clone())
    }
}

impl ToTokens for EnumOpts {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let EncodingUtilGeneric {
            typename,
            generics,
            where_clause,
            type_w_generics,
            type_params,
            allocator_bound,
            endecode_bound,
        } = self.into();

        let (impl_generics, _, _) = self.generics.split_for_impl();
        let variants = &self.data.as_enum().unwrap();
        let variants_to_u8 = variants
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let i = u8::try_from(i).expect("enum must have at most 255 variants");
                quote! { #i }
            })
            .collect::<Vec<_>>();
        let variant_idents = variants.iter().map(|v| v.ident.clone()).collect::<Vec<_>>();
        let variant_field_names = variants
            .iter()
            .map(|v| v.fields.field_names())
            .collect::<Vec<_>>();
        let destructures = variants
            .iter()
            .enumerate()
            .map(|(i, v)| v.destructure(Some(&variant_field_names[i])))
            .collect::<Vec<_>>();
        let variants_encode = variants.iter().enumerate().filter_map(|(i, variant)| {
            let ident = &variant.ident;
            if variant.skip.is_some() {
                return None;
            }

            let field_names = &variant_field_names[i];
            let fields_encode = variant.fields.iter().enumerate().map(|(i, field)| {
                let mut field = field.clone();
                field.ident = Some(field_names[i].clone());
                field.to_tokens(Some(i))
            });
            let destructure = &destructures[i];
            Some(quote! {
                Self::#ident #destructure => {
                    #(#fields_encode)*
                }
            })
        });
        let alloc_bounds = variants.iter().fold(AllocBound::Clone, |bound, v| {
            if matches!(min_bound(v.fields.iter()), AllocBound::Copy) {
                AllocBound::Copy
            } else {
                bound
            }
        });
        let allocator_bound = quote! {
            #allocator_bound + #alloc_bounds
        };
        let alloc_fns = variants
            .iter()
            .map(|variant| {
                variant.fields.iter().map(|field| {
                    field
                        .alloc
                        .clone()
                        .and_then(|a| a.with)
                        .unwrap_or(format_ident!("new_in"))
                })
            })
            .collect::<Vec<_>>();
        let variant_allocs = variants.iter().enumerate().map(|(i, variant)| {
            let alloc_fns = alloc_fns[i].clone();
            match variant.fields.style {
                ast::Style::Tuple => {
                    let field_types = variant.fields.iter().map(|f| &f.ty);

                    quote! { ( #(<#field_types>::#alloc_fns(alloc.clone())),* ) }
                }
                ast::Style::Struct => todo!(),
                ast::Style::Unit => {
                    quote! {}
                }
            }
        });

        let fields_size = variants
            .iter()
            .filter_map(|variant| {
                let fields = &variant.fields;
                if variant.skip.is_some() {
                    return None;
                }
                Some(fields.iter().filter_map(|field| {
                    let ty = field.ty.clone();
                    if field.skip.is_some() {
                        return None;
                    }
                    Some(quote! {
                        <#ty as ::pchan_encode::Endecode<A>>::size()?
                    })
                }))
            })
            .flatten();
        let result = quote! {
            impl <A: #allocator_bound, #(#type_params)*> #type_w_generics #where_clause {
                #[doc(hidden)]
                #[allow(dead_code)]
                fn _variant(&self) -> u8 {
                    match self {
                        #(Self::#variant_idents #destructures => #variants_to_u8),*
                    }
                }
                fn _create(variant: u8, alloc: A) -> ::core::result::Result<#type_w_generics, ::pchan_encode::EndecodeError> {
                    use ::pchan_encode::Allocate;
                    use ::core::result::Result::*;

                    match variant {
                        #(#variants_to_u8 => {
                            Ok(Self::#variant_idents #variant_allocs)
                        },)*
                        tag => Err(::pchan_encode::EndecodeError::InvalidTag(tag, core::any::type_name::<Self>())),
                    }
                }
            }

            impl<A: #allocator_bound, #(#type_params)*> #endecode_bound<A> for #type_w_generics #where_clause {
                fn endecode(&mut self, sink: &mut impl ::pchan_encode::Stream, alloc: A) -> Result<(), ::pchan_encode::EndecodeError> {
                    let original_variant = self._variant();
                    let mut variant = original_variant;
                    sink.u8(&mut variant)?;
                    if variant != original_variant {
                        *self = Self::_create(variant, alloc.clone())?;
                    }

                    match self {
                        #(#variants_encode),*
                    };
                    Ok(())
                }
                fn size() -> Result<Option<usize>, ::pchan_encode::EndecodeError> {
                    let mut size = 0;
                    #(
                    match #fields_size {
                        Some(s) => size += s,
                        None => return Ok(None)
                    };
                    )*
                    Ok(Some(size))
                }
            }
        };
        tokens.extend(result);
    }
}

#[proc_macro_derive(Endecode, attributes(endecode))]
pub fn encoding(item: TokenStream) -> TokenStream {
    let args = syn::parse_macro_input!(item as DeriveInput);
    match NamedStructOpts::from_derive_input(&args) {
        Err(_) => {}
        Ok(args) => return args.into_token_stream().into(),
    }
    match EnumOpts::from_derive_input(&args) {
        Err(err) => panic!("oopsies: {err}"),
        Ok(args) => return args.to_token_stream().into(),
    }
    panic!("shape not supported");
}
