# Getting your Hugging Face token for SAM 3

PixelKit's auto-labelling runs on Meta's **SAM 3**, whose weights are
distributed on Hugging Face behind a license gate. That means a one-time,
free setup: accept Meta's license, create a read token, paste it into
PixelKit. Roughly two minutes.

## Steps

1. **Create a Hugging Face account** (free): <https://huggingface.co/join>.
2. **Accept the SAM 3 license**: open
   <https://huggingface.co/facebook/sam3>, sign in, and click the button
   in the "gated model" banner to request/accept access. Approval is
   instant for this repo.
3. **Create a token**: <https://huggingface.co/settings/tokens> →
   *Create new token* → type **Read** (a fine-grained token with
   "Read access to contents of all public gated repos you can access"
   also works). Copy the `hf_…` value.
4. **Paste it into PixelKit**: the first-run setup asks for it (or
   Settings → Hugging Face access later). PixelKit validates it live and
   then downloads the weights (~1.7 GB) into your workspace's `weights/`
   folder.

The token is stored in the app config directory on your machine —
never inside the workspace, so backups and synced workspaces don't
contain it. PixelKit uses it only to download weights from Hugging Face.

## If validation fails

- **"Token invalid"** — the value didn't authenticate: re-copy it
  (tokens start with `hf_`), or it may have been revoked.
- **"Token OK but no SAM 3 access"** — the token works but step 2 was
  skipped or done with a different account: open the
  [facebook/sam3](https://huggingface.co/facebook/sam3) page with the
  same account the token belongs to and accept the license.
- **Corporate networks** — downloads come from `huggingface.co` /
  `cdn-lfs.huggingface.co`; proxies that block them will fail the
  download step, not the token check.

## Community mirrors

`SAM3_MODEL_ID` (env var) overrides the repo id if you want to use a
compatible repackage of the weights. The default and only tested source
is `facebook/sam3`.

## DINOv2

The second model PixelKit uses, `facebook/dinov2-large` (~0.6 GB), is
not gated — it downloads automatically without any token.
