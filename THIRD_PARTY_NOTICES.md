# Third-Party Notices

このファイルは、RAVE Web Research が使用・改変するモデル、モデル変換用ソース、ブラウザー実行環境の出典と条件を記録します。本リポジトリーの独自部分に対するルートの [LICENSE](LICENSE) は、以下の第三者の権利・ライセンスを置き換えるものではありません。Release assets に含まれる個別の attribution、license、notice も保持してください。

## 1. RAVE の研究とソース

RAVE の研究・アーキテクチャーの著者は **Antoine Caillon and Philippe Esling / ACIDS–IRCAM** です。

- 論文：[RAVE: A variational autoencoder for fast and high-quality neural audio synthesis](https://arxiv.org/abs/2111.05011)
- 研究プロジェクト：[ACIDS–IRCAM / RAVE](https://github.com/acids-ircam/RAVE)

Organ の変換では、次の固定版のソースを使用します。

| ソース | 固定コミット | ライセンス |
| --- | --- | --- |
| [Victor Shepardson / RAVE](https://github.com/victor-shepardson/RAVE/tree/ab7d40d6740130f0268debbc4b6633318357ea7d) | `ab7d40d6740130f0268debbc4b6633318357ea7d` | [CC BY-NC 4.0・現物 LICENSE](https://github.com/victor-shepardson/RAVE/blob/ab7d40d6740130f0268debbc4b6633318357ea7d/LICENSE) |
| [Victor Shepardson / cached_conv](https://github.com/victor-shepardson/cached_conv/tree/d2da85992d030dcf1dd8ebf318c3d69753729295) | `d2da85992d030dcf1dd8ebf318c3d69753729295` | [CC BY-NC 4.0・現物 LICENSE](https://github.com/victor-shepardson/cached_conv/blob/d2da85992d030dcf1dd8ebf318c3d69753729295/LICENSE) |

変換スクリプトは元ソースの causal padding、状態、ノイズ、保存済みの投影を参照しています。別の同名パッケージや、現行の main ブランチでの再現を主張するものではありません。再取得する際も固定コミットとそのライセンスを使用してください。

## 2. IIL Organ checkpoint と改変 ONNX

**提供元：Intelligent Instruments Lab（IIL）**。モデルの元となる RAVE の著者、および使用する causal fork の作者は前節を参照してください。

- 固定モデルリポジトリー：[`Intelligent-Instruments-Lab/rave-models` at `c25a03d625840c40cd3a48779ed72f2a1947d7b4`](https://huggingface.co/Intelligent-Instruments-Lab/rave-models/tree/c25a03d625840c40cd3a48779ed72f2a1947d7b4/checkpoints/organ_archive_b512_r48000)
- checkpoint：[`organ_archive_b512_r48000/version_1/checkpoints/last.ckpt`](https://huggingface.co/Intelligent-Instruments-Lab/rave-models/resolve/c25a03d625840c40cd3a48779ed72f2a1947d7b4/checkpoints/organ_archive_b512_r48000/version_1/checkpoints/last.ckpt)
- 設定：[`organ_archive_b512_r48000/config.gin`](https://huggingface.co/Intelligent-Instruments-Lab/rave-models/raw/c25a03d625840c40cd3a48779ed72f2a1947d7b4/checkpoints/organ_archive_b512_r48000/config.gin)
- ライセンスの根拠：[固定版の IIL モデルカード](https://huggingface.co/Intelligent-Instruments-Lab/rave-models/blob/c25a03d625840c40cd3a48779ed72f2a1947d7b4/README.md)
- ライセンス：[Creative Commons Attribution–NonCommercial 4.0 International（CC BY-NC 4.0）](https://creativecommons.org/licenses/by-nc/4.0/)

元ファイルの SHA-256：

```text
last.ckpt   218518cb0f7c2670a9cd3973c3e2972632893f599deacc851998b372a8f8a22e
config.gin  975e0e8615204af6c9f8aa901275ac634d70f60011c898dbc103e1e4fb09ccb0
```

本プロジェクトでの変更（2026-10-26）：学習済みの encoder / decoder を ONNX へ書き出し、36 個の過去状態と変分サンプリングの正規ノイズを明示的な入出力にしました。128 次元の posterior を保持し、checkpoint に保存された PCA のうち 6 方向へ gain / bias の差分操作を加えました。推論用に weight normalization を畳み込んでいます。再学習や重みの置き換えは行っていません。

Organ 1024 は同じ学習済みの重みで 512 サンプルの処理ステップを 2 回まとめる版です。独立して再学習したモデルではありません。これらの改変モデルも CC BY-NC 4.0 の条件で配布します。

この出典は **b512 の checkpoint** です。別途公開された b2048 の Organ TorchScript と同じファイルではありません。b2048 の説明から、この checkpoint の正確な学習録音一覧を推定していません。

## 3. Water / Pond Brain と改変 ONNX

**提供元：Intelligent Instruments Lab（IIL）**。Jenna Sutela の *Pond Brain* のために制作された Water モデルです。上流の説明・録音提供者の表示は [固定版 IIL モデルカードの Pond Brain Marine Sounds 節](https://huggingface.co/Intelligent-Instruments-Lab/rave-models/blob/c25a03d625840c40cd3a48779ed72f2a1947d7b4/README.md#pond-brain-marine-sounds) を参照してください。

- 上流でのモデル名：`water_pondbrain_b2048_r48000_z16.ts`
- 上流のモデルカードのライセンス：**CC BY-NC 4.0**
- 本プロジェクトが直接改変したもの：既存のブラウザー実験で使用した Water / IIL / Pond Brain の stateful ONNX。使用した中間ファイルは下記の SHA-256 で識別します。
- 中間 ONNX：103,384,587 bytes、198 個の分割ファイルから復元

中間 ONNX の SHA-256：

```text
b6c84085f475b3bddf69eb2ee0a85836230e0462d7d112cc3a2cb6e497914e47
```

本プロジェクトでの変更（2026-10-26）：保持された潜在表現の境界に float32 の `latent_gain` / `latent_bias` 入力を追加しました。2048 版ではブロック長に依存する 2 個の定数と shape 宣言を変更しています。学習済みの重みは再学習・置換していません。保持された 16 個の潜在座標のうち 6 個を画面で操作します。これを PCA や意味の決まった座標とは説明していません。

上流の TorchScript と、直接使用した中間 ONNX は別の成果物です。本プロジェクトは元の TorchScript → ONNX 変換器を再現したものではなく、両者の数値的な同一性も主張していません。改変の数値検証は上記の中間 ONNX を基準にしています。

改変 Water モデルにも CC BY-NC 4.0 の条件が適用されます。上流のモデルカードには、Pond Brain decoder の初期化時に大きな過渡音が出る場合があるとの注意があります。

## 4. Djembe / Scyclone と改変 ONNX

**提供元：Torsion Audio**。RAVE の研究・アーキテクチャーの著者は Antoine Caillon and Philippe Esling / ACIDS–IRCAM です。

- 元モデル：[`Torsion-Audio/Scyclone/assets/models/djembe.onnx`](https://github.com/Torsion-Audio/Scyclone/blob/2b1990c4ad04fec1981d21bdde94fdafbadb84e0/assets/models/djembe.onnx)
- 固定コミット：`2b1990c4ad04fec1981d21bdde94fdafbadb84e0`
- 元ファイル：16,622,944 bytes
- モデルのライセンスの根拠：[固定版 Scyclone README の Licenses 節](https://github.com/Torsion-Audio/Scyclone/blob/2b1990c4ad04fec1981d21bdde94fdafbadb84e0/README.md#licenses)
- モデルのライセンス：**CC BY-NC 4.0**

元 ONNX の SHA-256：

```text
ea94557a8d9b40f22a1dadbc93ab98f46144a68ce0bed1403a1558a2d1007f16
```

本プロジェクトでの変更（2026-10-26）：正規ノイズを入力化し、畳み込みの過去状態と転置畳み込みのキャッシュ・出力シフトを追加しました。合成時の位相を調整し、元から保持された 4 個の投影座標へ gain / bias 操作を加えています。元の 46 個の initializer は保持し、再学習していません。

ストリーミング化による既知の追加シフトは 1,360 サンプルです。元モデル自体の絶対的な入出力のずれは確認できていません。正規ノイズを与えた比較には数値的な残差があり、この版は実験的な比較対象です。改変モデルにも CC BY-NC 4.0 の条件が適用されます。Scyclone アプリケーションのコード全体を、本リポジトリーのライセンスで再配布するものではありません。

## 5. ONNX Runtime Web

**Copyright (c) Microsoft Corporation.** ONNX Runtime Web は **MIT License** です。モデルの CC BY-NC 4.0 と、実行環境の MIT License は別々に適用されます。

- 配布パッケージ：`onnxruntime-web` **1.30.0**
- 取得元：[npm の固定版 tarball](https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-1.30.0.tgz)
- 公式ソースのライセンス：[Microsoft ONNX Runtime v1.30.0 LICENSE](outputs/rave-morph/web/vendor/ort/LICENSE)
- npm integrity（取得時に照合）：

```text
sha512-q0y+JrrtukXSzsBWEMccVfqX25LRmosXHF+CaRJmg8pZClzcV7svNc4rKY3jL02Vb7QmRMDs1SigqR4CXAfKYQ==
```

ブラウザー用 JS / MJS / WASM とともに、パッケージ内の **`LICENSE` と `ThirdPartyNotices.txt` の両方を保持**してください。後者には Runtime に含まれる依存物の表示があります。Release assets から復元される `web/vendor/ort/` 内にもこれらを含めます。

モデル変換・検証用の Python 依存関係はブラウザー配布物とは別です。たとえば model-tools の native `onnxruntime` とブラウザー用 `onnxruntime-web` は別パッケージです。依存関係を取得・再配布する場合は、それぞれの配布物に付属するライセンスを保持してください。

## 6. 非商用条件と再配布

モデルおよび CC BY-NC 4.0 が適用される改変部分を共有する際は、少なくとも次を引き継いでください。

- 作者・提供元と、上記の出典リンク。
- CC BY-NC 4.0 の表示とリンク、および元から付属する権利・免責の表示。
- 本プロジェクトによる変更の表示。さらに変更した場合は、その変更も示してください。
- 非商用利用という条件。

ONNX など別のファイル形式にしたこと、ブラウザーへ配信したこと、ソースを公開したことだけでは非商用条件は解除されません。CC BY-NC 4.0 の正式な範囲・条件は [Legal Code](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en) に従います。本プロジェクトは商用利用の追加許諾を取得・付与していません。

ONNX Runtime とその依存物のように別のライセンスが付く作品には、その条件を適用してください。本プロジェクトの CC BY-NC 表示を、それらの第三者の作品に対する追加制限として扱わないでください。

本プロジェクトは独立した研究実装です。IIL、ACIDS–IRCAM、Victor Shepardson、Jenna Sutela、Torsion Audio、Microsoft、Neutone などの作者・組織による承認や支援を示すものではありません。Neutone Morpho のエンジン・モデル・バイナリーは含まれていません。
