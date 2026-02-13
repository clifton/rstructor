use rstructor::MediaFile;

pub const RUST_LOGO_URL: &str = "https://www.rust-lang.org/logos/rust-logo-512x512.png";
pub const RUST_LOGO_MIME: &str = "image/png";

#[allow(dead_code)]
pub const RUST_SOCIAL_URL: &str = "https://www.rust-lang.org/static/images/rust-social-wide.jpg";
#[allow(dead_code)]
pub const RUST_SOCIAL_MIME: &str = "image/jpeg";

pub async fn download_media(url: &str, mime: &str) -> MediaFile {
    let bytes = reqwest::get(url)
        .await
        .expect("Failed to download media fixture")
        .bytes()
        .await
        .expect("Failed to read media fixture bytes");
    MediaFile::from_bytes(&bytes, mime)
}

#[allow(dead_code)]
pub fn media_url(url: &str, mime: &str) -> MediaFile {
    MediaFile::new(url, mime)
}
