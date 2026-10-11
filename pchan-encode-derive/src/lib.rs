extern crate alloc;

use alloc::borrow::Cow;

use darling::{FromDeriveInput, FromField, FromMeta, FromVariant, ast};
use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::DeriveInput;

#[derive(Debug, FromDeriveInput)]
#[darling(
    attributes(endecode),
    supports(struct_named, struct_tuple),
    forward_attrs(allow, doc, cfg)
)]
struct NamedStructOpts {
    ident:    syn::Ident,
    generics: syn::Generics,
    data:     ast::Data<(), EncodingField>,
    alloc:    Option<AllocInner>,
}

#[derive(Debug, FromField, Clone)]
#[darling(attributes(endecode), forward_attrs(allow, doc, cfg))]
struct EncodingField {
    ident:  Option<syn::Ident>,
    skip:   Option<()>,
    ty:     syn::Type,
    alloc:  Option<Alloc>,
    string: Option<()>,
}

#[derive(Debug, FromMeta, Clone)]
struct Alloc {
    with:  Option<syn::Ident>,
    #[darling(flatten)]
    inner: AllocInner,
}

#[derive(Debug, FromMeta, Clone)]
struct AllocInner {
    bounds:  Option<AllocBound>,
    /// ## Container Attribute
    /// generates a default [`Allocate`] implementation, providing a `new_in` method.
    /// ## Struct Field Attribute
    /// uses [`Default`] trait for [`Allocate`] implementation.
    /// ## Enum Variant Attribute
    /// marks this variant as default
    default: Option<()>,
    global:  Option<()>,
}

#[derive(Default, Debug, Clone, FromMeta)]
enum AllocBound {
    #[default]
    Clone,
    Copy,
}

enum AllocCall {
    Default,
    Allocate,
    Custom(syn::Ident),
    NoAlloc(syn::Ident),
}

fn default_endecode_bound() -> proc_macro2::TokenStream {
    quote! {
        ::pchan_encode::Endecode<A>
    }
}

impl AllocInner {
    fn alloc_call(&self, string: Option<()>) -> AllocCall {
        if string.is_some() {
            return AllocCall::NoAlloc(format_ident!("new"));
        }
        match self.default {
            Some(()) => AllocCall::Default,
            None => AllocCall::Allocate,
        }
    }
    fn endecode_bound(&self) -> proc_macro2::TokenStream {
        match self.global {
            Some(_) => quote! { ::pchan_encode::Endecode<alloc::alloc::Global> },
            None => default_endecode_bound(),
        }
    }
}

impl Alloc {
    fn alloc_call(&self, string: Option<()>) -> AllocCall {
        if let Some(with) = &self.with {
            return AllocCall::Custom(with.clone());
        };
        self.inner.alloc_call(string)
    }
}

fn min_bound<'a>(fields: impl IntoIterator<Item = &'a EncodingField>) -> AllocBound {
    let copy = fields.into_iter().any(|field| {
        field
            .alloc
            .as_ref()
            .is_some_and(|a| matches!(a.inner.bounds, Some(AllocBound::Copy)))
    });
    if copy {
        AllocBound::Copy
    } else {
        AllocBound::Clone
    }
}

impl AllocCall {
    fn to_tokens(
        &self,
        ty: &syn::Type,
        allocator: proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream {
        match self {
            AllocCall::Default => quote! { <#ty as Default>::default() },
            AllocCall::Allocate => {
                quote! {
                    {
                        use ::pchan_encode::Allocate;
                        <#ty>::new_in(#allocator)
                    }
                }
            }
            AllocCall::Custom(ident) => quote! {
                <#ty>::#ident(#allocator)
            },
            AllocCall::NoAlloc(ident) => quote! {
                <#ty>::#ident()
            },
        }
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
    where_clause:    Option<syn::WhereClause>,
    type_w_generics: proc_macro2::TokenStream,
    type_params:     Vec<proc_macro2::TokenStream>,
    allocator_bound: proc_macro2::TokenStream,
    endecode_bound:  proc_macro2::TokenStream,
    has_alloc:       bool,
}

impl EncodingUtilGeneric {
    fn from_ident_and_generics(ident: syn::Ident, generics: syn::Generics) -> Self {
        let typename = ident.clone();
        let generics = generics.clone();
        let (_, type_generics, where_c) = generics.split_for_impl();
        let where_clause = where_c.cloned();
        let mut has_alloc = false;
        let type_params = generics
            .type_params()
            .filter_map(|ty| {
                let ident = &ty.ident;
                if &ident.to_string() == "A" {
                    has_alloc = true;
                    return None;
                }
                let bounds = &ty.bounds;
                match bounds.len() {
                    0 => Some(quote! {
                        #ident :   ::pchan_encode::Endecode<A>
                    }),
                    _ => Some(quote! {
                        #ident : #bounds + ::pchan_encode::Endecode<A>
                    }),
                }
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
            where_clause,
            type_w_generics,
            type_params,
            allocator_bound,
            endecode_bound,
            has_alloc,
        }
    }
}

impl From<&NamedStructOpts> for EncodingUtilGeneric {
    fn from(value: &NamedStructOpts) -> Self {
        Self::from_ident_and_generics(value.ident.clone(), value.generics.clone())
    }
}

fn allocator_param(alloc: Option<&Alloc>, string: Option<()>) -> proc_macro2::TokenStream {
    match alloc.as_ref().and_then(|a| a.inner.global).or(string) {
        Some(_) => quote! {alloc::alloc::Global},
        None => {
            quote! { alloc.clone() }
        }
    }
}

fn allocator_param_inner(
    alloc: Option<&AllocInner>,
    string: Option<()>,
) -> proc_macro2::TokenStream {
    match alloc.and_then(|a| a.global).or(string) {
        Some(_) => quote! {alloc::alloc::Global},
        None => {
            quote! { alloc.clone() }
        }
    }
}

impl EncodingField {
    fn to_tokens(&self, index: Option<usize>) -> Option<proc_macro2::TokenStream> {
        let ident = &self.ident;
        if self.skip.is_some() {
            return None;
        }
        let allocator = allocator_param(self.alloc.as_ref(), self.string);
        match ident {
            Some(ident) => Some(quote! {
                #ident.endecode(sink, #allocator)?;
            }),
            None => {
                let ident = syn::Index::from(index.expect("field with no name must be provided an index. this is a bug in the derive proc macro."));
                Some(quote! {
                    #ident.endecode(sink, #allocator)?;
                })
            }
        }
    }
}

struct FieldSetEncode<'a> {
    fields: &'a [EncodingField],
}

impl ToTokens for FieldSetEncode<'_> {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let fields_encode = self
            .fields
            .iter()
            .enumerate()
            .filter_map(|(i, field)| field.to_tokens(Some(i)));
        let res = quote! {
            #(self. #fields_encode)*
        };
        tokens.extend(res);
    }
}

impl ToTokens for NamedStructOpts {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let EncodingUtilGeneric {
            where_clause,
            type_w_generics,
            type_params,
            allocator_bound,
            endecode_bound: _,
            has_alloc: _,
        } = self.into();

        let const_params = self.generics.const_params().collect::<Vec<_>>();

        let fields = &self.data.as_struct().unwrap().fields;
        let fields_set = FieldSetEncode { fields };
        let _container_endecode_bound = self
            .alloc
            .as_ref()
            .map_or_else(default_endecode_bound, |a| a.endecode_bound());
        let fields_size = fields.iter().filter_map(|field| {
            let ty = &field.ty;
            if field.skip.is_some() {
                return None;
            }
            let field_endecode_bound = field
                .alloc
                .as_ref()
                .map_or_else(default_endecode_bound, |a| a.inner.endecode_bound());
            Some(quote! {
                <#ty as #field_endecode_bound>::size()?
            })
        });

        let allocator = allocator_param_inner(self.alloc.as_ref(), None);
        let fields_init = fields.iter().map(|field| {
            let name = &field.ident;
            let ty = &field.ty;
            let alloc_call = field.alloc.as_ref().map_or(
                field.string.map_or(AllocCall::Default, |_| {
                    AllocCall::NoAlloc(format_ident!("new"))
                }),
                |a| a.alloc_call(field.string),
            );
            let func_call = alloc_call.to_tokens(ty, allocator.clone());
            quote! {
                #name: #func_call
            }
        });

        let mut result = quote! {
            impl<#(#const_params,)* A: #allocator_bound + Clone, #(#type_params)*> ::pchan_encode::Endecode<A> for #type_w_generics #where_clause {
                fn endecode(&mut self, sink: &mut impl ::pchan_encode::Stream, alloc: A) -> Result<(), ::pchan_encode::EndecodeError> {
                    #fields_set
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

        if self.alloc.as_ref().is_some_and(|a| a.default.is_some()) {
            let _allocator = allocator_param_inner(self.alloc.as_ref(), None);
            result = quote! {
                #result
                impl<#(#const_params,)* A: #allocator_bound + Clone, #(#type_params)*> ::pchan_encode::Allocate<A> for #type_w_generics #where_clause {
                    fn new_in(alloc: A) -> #type_w_generics {
                        Self {
                            #(#fields_init),*
                        }
                    }
                }
            };
        }

        tokens.extend(result);
    }
}

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(endecode), supports(enum_any))]
struct EnumOpts {
    ident:    syn::Ident,
    generics: syn::Generics,
    data:     ast::Data<EncodingVariant, ()>,
    alloc:    Option<Alloc>,
}

#[derive(Debug, FromVariant)]
#[darling(attributes(endecode))]
struct EncodingVariant {
    ident:  syn::Ident,
    skip:   Option<()>,
    fields: ast::Fields<EncodingField>,
    alloc:  Option<AllocInner>,
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
        match self.fields.style {
            ast::Style::Tuple => quote! { (#(#field_names),*) },
            ast::Style::Struct => quote! { {#(#field_names),*} },
            ast::Style::Unit => quote! {},
        }
    }

    fn destructure_ignore(&self, field_names: Option<&[syn::Ident]>) -> proc_macro2::TokenStream {
        let field_names = match field_names {
            Some(idents) => Cow::Borrowed(idents),
            None => Cow::Owned(self.fields.field_names()),
        };
        let count = field_names.len();
        let underscores = (0..count).map(|_| quote! { _ });
        match self.fields.style {
            ast::Style::Tuple => quote! { (#(#underscores),*) },
            ast::Style::Struct => quote! { { .. } },
            ast::Style::Unit => quote! {},
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
            where_clause,
            type_w_generics,
            type_params,
            allocator_bound,
            endecode_bound,
            has_alloc,
        } = self.into();

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
        let ignored_destructures = variants
            .iter()
            .enumerate()
            .map(|(i, v)| v.destructure_ignore(Some(&variant_field_names[i])))
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
                        .as_ref()
                        .map_or(AllocCall::Default, |a| a.alloc_call(field.string))
                })
            })
            .collect::<Vec<_>>();
        let variant_allocs = variants
            .iter()
            .enumerate()
            .map(|(i, variant)| {
                let field_allocations = variant
                    .fields
                    .iter()
                    .map(|f| (&f.ty, allocator_param(f.alloc.as_ref(), f.string)));
                let field_names = &variant_field_names[i];
                let alloc_fns = alloc_fns[i]
                    .clone()
                    .zip(field_allocations)
                    .map(|(call, (ty, allocator))| call.to_tokens(ty, allocator));
                match variant.fields.style {
                    ast::Style::Tuple => {
                        quote! { ( #(#alloc_fns),* ) }
                    }
                    ast::Style::Struct => {
                        quote! { { #(#field_names: #alloc_fns),* } }
                    }
                    ast::Style::Unit => {
                        quote! {}
                    }
                }
            })
            .collect::<Vec<_>>();

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
        let allocator_type_param = if has_alloc {
            Some(quote! {A:#allocator_bound,})
        } else {
            None
        };
        let (create_fn_sig_allocator_param, create_fn_allocator_param) = if has_alloc {
            (Some(quote! {alloc: A}), Some(quote! { alloc.clone()  }))
        } else {
            (None, None)
        };
        let mut result = quote! {
            impl <#allocator_type_param #(#type_params)*> #type_w_generics #where_clause {
                #[doc(hidden)]
                #[allow(dead_code)]
                fn _variant(&self) -> u8 {
                    match self {
                        #(Self::#variant_idents #ignored_destructures => #variants_to_u8),*
                    }
                }
                fn _create(variant: u8, #create_fn_sig_allocator_param) -> ::core::result::Result<#type_w_generics, ::pchan_encode::EndecodeError> {
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
                        *self = Self::_create(variant, #create_fn_allocator_param)?;
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

        if let Some(alloc) = self.alloc.as_ref()
            && alloc.inner.default.is_some()
        {
            let default_variant = variants
                .iter()
                .enumerate()
                .filter(|(_, v)| v.alloc.as_ref().is_some_and(|a| a.default.is_some()))
                .collect::<Vec<_>>();
            assert!(
                default_variant.len() <= 1,
                "Must have at most one variant marked as default."
            );
            let (i, default_variant) = *default_variant
                .first()
                .expect("Must have at least one variant marked as default");
            let ident = &default_variant.ident;
            let alloc = &variant_allocs[i];

            result = quote! {
                #result
                impl<A: #allocator_bound, #(#type_params)*> ::pchan_encode::Allocate<A> for #type_w_generics #where_clause {
                    fn new_in(alloc: A) -> #type_w_generics {
                        Self::#ident #alloc
                    }
                }
            }
        }

        tokens.extend(result);
    }
}

/// # Attributes
///
/// ## `alloc`
///
/// - container: generates a default [`Allocate`] implementation, providing a `new_in` method.
/// - field: uses [`Default`] trait for [`Allocate`] implementation.
/// - variant: marks this variant as default
#[proc_macro_derive(Endecode, attributes(endecode))]
pub fn derive_endecode(item: TokenStream) -> TokenStream {
    let args = syn::parse_macro_input!(item as DeriveInput);
    match NamedStructOpts::from_derive_input(&args) {
        Err(_) => {}
        Ok(args) => return args.into_token_stream().into(),
    }
    match EnumOpts::from_derive_input(&args) {
        Err(_) => {}
        Ok(args) => return args.to_token_stream().into(),
    }
    panic!("shape not supported");
}
