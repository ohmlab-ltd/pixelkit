# Hugging Face token

SAM 3, the model behind auto-labelling, is distributed by Meta on Hugging
Face behind a licence agreement. Downloading it needs a free Hugging Face
account that has accepted the licence, and a read token from that account.

## Steps

1. Create an account at <https://huggingface.co/join>.
2. Sign in, open <https://huggingface.co/facebook/sam3> and accept the
   licence in the gated-model banner.
3. Create a token at <https://huggingface.co/settings/tokens> of type Read.
   A fine-grained token with "Read access to contents of all public gated
   repos you can access" also works. Tokens start with `hf_`.
4. Enter the token in PixelKit, in the first-run setup or under Settings,
   Hugging Face access. PixelKit checks it, then downloads the SAM 3 weights,
   about 3.6 GB, into the workspace's `weights/` folder. Without a GPU, the
   download starts after Compute is set to CPU in Settings and PixelKit
   restarts.

The token is stored in the application's config folder, not in the
workspace, and is used only to download weights from Hugging Face.

## Errors

- A token that does not authenticate is rejected. It may be mistyped or
  revoked.
- A token that authenticates without SAM 3 access belongs to an account that
  has not accepted the licence on the facebook/sam3 page.
- A proxy that blocks `huggingface.co` fails the token check. One that lets
  the check through but blocks the file downloads fails the download.

## Other sources

The `SAM3_MODEL_ID` environment variable replaces the repository id, for a
compatible repackage of the weights. The default, and the only tested source,
is `facebook/sam3`.

## DINOv2

`facebook/dinov2-large`, about 1.3 GB, is not gated and downloads without a
token.
