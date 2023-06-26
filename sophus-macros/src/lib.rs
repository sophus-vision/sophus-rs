extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream, Result},
    punctuated::Punctuated,
    token::Comma,
    Ident,
};

struct Idents(Punctuated<Ident, Comma>);

impl Parse for Idents {
    fn parse(input: ParseStream) -> Result<Self> {
        let parsed = Punctuated::parse_terminated(input)?;
        Ok(Idents(parsed))
    }
}
fn create_output_tuple<'a>(idents: impl Iterator<Item = &'a Ident>) -> proc_macro2::TokenStream {
    let mut output_tuple = quote! { () };
    let mut tuple_elements: Vec<_> = idents.map(|ident| quote! { #ident::Arg }).collect();
    tuple_elements.reverse();
    for element in tuple_elements {
        output_tuple = quote! { (#element, #output_tuple) };
    }
    output_tuple
}

fn create_cat_array<'a>(idents: impl Iterator<Item = &'a Ident>) -> proc_macro2::TokenStream {
    let mut cat_array = quote! {};
    let cat_elements: Vec<_> = idents.map(|ident| quote! { #ident::CAT }).collect();
    for element in cat_elements {
        cat_array = quote! { #cat_array #element ,};
    }
    cat_array
}

fn create_fn_body<'a>(idents: impl Iterator<Item = &'a Ident>) -> proc_macro2::TokenStream {
    let mut fn_body = quote! { () };

    let idents: Vec<_> = idents.collect();

    for (j, _) in idents.iter().enumerate().rev() {
        let i = syn::Index::from(j);
        fn_body = quote! { (self.#i.get_elem(idx[#i]), #fn_body) };
    }

    fn_body
}

#[proc_macro]
pub fn create_impl(input: TokenStream) -> TokenStream {
    let idents = syn::parse_macro_input!(input as Idents);

    let mut expanded = quote! {
            impl<M0: ManifoldV> ManifoldVTuple for (M0)
            where
                Self: Sized,
            {
                type Idx = [usize; 1usize];
                type Output = (M0::Arg, ());
                type CatArray = [char; 1usize];
                const CAT: Self::CatArray = [M0::CAT];
                fn get_elem(&self, idx: &Self::Idx) -> Self::Output {
                    (self.get_elem(idx[0]), ())
                }
            }
        };

    for n in 2..=idents.0.len() {
        let tuple_idents = idents.0.iter().take(n);
        let idx_len = n;

        let output_tuple = create_output_tuple(tuple_idents.clone());
        let cat_array = create_cat_array(tuple_idents.clone());
        let fn_body = create_fn_body(tuple_idents.clone());

        let tuple_idents_0 = tuple_idents.clone();

        expanded = quote! {
            #expanded

            impl<#(#tuple_idents_0: ManifoldV),*> ManifoldVTuple for (#(#tuple_idents),*)
            where
                Self: Sized,
            {
                type Idx = [usize; #idx_len];
                type Output = #output_tuple;
                type CatArray = [char; #idx_len];
                const CAT: Self::CatArray = [#cat_array];

                fn get_elem(&self, idx: &Self::Idx) -> Self::Output {
                    #fn_body
                }
            }
        };
    }

    expanded.into()
}
