#![feature(proc_macro_totokens)]
extern crate proc_macro;
use darling::*;
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
        }
    }
}

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(encode), supports(struct_named))]
struct EncodingOpts {
    ident:  syn::Ident,
    opcode: syn::Expr,
    funct:  Option<syn::Expr>,
    rt:     Option<syn::Expr>,
    rs:     Option<syn::Expr>,
    order:  Option<EncodeParamOrder>,
    cop:    Option<syn::Expr>,
    data:   ast::Data<(), EncodingField>,
}

#[derive(Debug, FromMeta)]
enum EncodeParamOrder {
    RtRs,
    RsRt,
}

#[derive(Debug, FromField)]
#[darling(attributes(encode))]
struct EncodingField {
    ident: Option<syn::Ident>,
    ty:    syn::Type,
}

#[proc_macro_derive(Encode, attributes(encode))]
pub fn encoding(item: TokenStream) -> TokenStream {
    let args = syn::parse_macro_input!(item as DeriveInput);
    let args = match EncodingOpts::from_derive_input(&args) {
        Err(e) => panic!("{e}"),
        Ok(args) => args,
    };
    let EncodingOpts {
        ident,
        funct,
        opcode: prime_op,
        rt: rt_const,
        cop: cop_const,
        rs: rs_const,
        order,
        data,
    } = args;

    let data = data.take_struct().unwrap();

    let find_field_with_attr = |search_ident| {
        data.fields.iter().find(|field| {
            field
                .ident
                .as_ref()
                .map(|ident| *ident == search_ident)
                .unwrap_or(false)
        })
    };

    let ident_lowercase = ident.to_string().to_lowercase();
    let ident_lowercase = syn::Ident::from_string(&ident_lowercase).unwrap();

    let rd = find_field_with_attr("rd");
    let rt = find_field_with_attr("rt");
    let rs = find_field_with_attr("rs");
    let cop = find_field_with_attr("cop");
    let imm16 = find_field_with_attr("imm16");
    let imm26 = find_field_with_attr("imm26");
    let shamt = find_field_with_attr("shamt");

    let set_rd = rd.map(|_| quote! { .with_rd(u5::new(self.rd as _)) });
    let set_rt = rt.map(|_| quote! { .with_rt(u5::new(self.rt as _)) });
    let set_rt_const = rt_const.map(|rt| quote! { .with_rt(u5::new(#rt)) });
    let set_rs = rs.map(|_| quote! { .with_rs(u5::new(self.rs as _)) });
    let set_rs_const = rs_const.map(|rs| quote! { .with_rs(u5::new(#rs)) });
    let set_cop = cop.map(|_| quote! { .with_cop(u2::new(self.cop as _)) });
    let set_cop_const = cop_const.map(|cop| quote! { .with_cop(u2::new(#cop as _)) });
    let set_funct = funct
        .as_ref()
        .map(|_| quote! { .with_funct(u6::new(#funct as _)) });
    let set_shamt = shamt
        .as_ref()
        .map(|_| quote! { .with_shamt(u5::new(self.shamt as _)) });
    let set_prime_op = quote! { .with_opcode(u6::new(#prime_op as _)) };

    let set_imm = match (imm16, imm26) {
        (Some(_), None) => quote! { .with_imm16(self.imm16 as _) },
        (None, Some(_)) => quote! { .with_imm26(u26::new(self.imm26 as _)) },
        (None, None) => quote! {},
        (Some(_), Some(_)) => panic!("imm16 and imm26 fields cannot be specified together."),
    };

    let into_impl = quote! {
        impl const Into<OpCode> for #ident {
            fn into(self: Self) -> OpCode {
                OpCode::default()
                    #set_prime_op
                    #set_funct
                    #set_rd
                    #set_rt
                    #set_rt_const
                    #set_rs
                    #set_rs_const
                    #set_cop
                    #set_cop_const
                    #set_imm
                    #set_shamt
            }
        }
    };

    // let quote_ty = |field: &EncodingField| {
    //     let ty = &field.ty;
    //     quote! { #ty }
    // };

    let order = match order {
        Some(order) => order,
        None => match (rd, imm16) {
            (None, Some(_)) => EncodeParamOrder::RtRs,
            (Some(_), None) => EncodeParamOrder::RsRt,
            (None, None) => EncodeParamOrder::RsRt,
            (Some(_), Some(_)) => panic!("rd field cannot be specified with imm"),
        },
    };

    let param_iter = match cop {
        None => [cop, rd]
            .into_iter()
            .chain(match order {
                EncodeParamOrder::RtRs => [rt, rs].into_iter(),
                EncodeParamOrder::RsRt => [rs, rt].into_iter(),
            })
            .chain([imm16, imm26, shamt])
            .flatten()
            .collect::<Vec<_>>(),
        Some(_) => [cop]
            .into_iter()
            .chain(match order {
                EncodeParamOrder::RtRs => [rt, rs].into_iter(),
                EncodeParamOrder::RsRt => [rs, rt].into_iter(),
            })
            .chain([rd, imm16, imm26, shamt])
            .flatten()
            .collect::<Vec<_>>(),
    };

    let args = param_iter
        .iter()
        .map(|field| {
            let ident = &field.ident;
            let ty = &field.ty;
            quote! { #ident: #ty }
        })
        .collect::<Vec<_>>();

    let values = param_iter
        .iter()
        .map(|field| {
            let ident = &field.ident;
            quote! { #ident }
        })
        .collect::<Vec<_>>();

    // let types = param_iter.clone().map(quote_ty).collect::<Vec<_>>();

    let new_func = quote! {
        impl #ident {
            #[inline(always)]
            pub const fn new(#(#args),*) -> Self {
                Self {
                    #(#values),*
                }
            }
        }
        #[inline(always)]
        pub const fn #ident_lowercase(#(#args),*) -> OpCode {
            #ident::new(#(#values),*).into()
        }
    };

    quote! {
        #into_impl
        #new_func
    }
    .into()
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
                        .map(|ident| ident == "default")
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

#[proc_macro_attribute]
pub fn instrument_write(_attr: TokenStream, item: TokenStream) -> TokenStream {
    format!(
        r#"#[instrument(
            level = Level::TRACE,
            skip(self, address),
            fields(
                address = %hex(address),
                value = %hex(value),
                isc = unsafe {{ (*self).cpu.isc() }}
            )
        )]
        {}
        "#,
        item
    )
    .parse()
    .expect("invalid tokens")
}
