# RAVE Web

学習済みの RAVE モデルで、入力した音の音色を変える研究用プロトタイプです。**自分のPCで起動し、ブラウザー内で音を処理します。非商用利用向けのソース配布です。**

## 自分のPCで試す

1. **Python 3.9以上**を [公式サイト](https://www.python.org/downloads/) からインストールします。すでに入っていれば不要です。
2. [制作コードのZIP](https://github.com/khrtkn/rave-web/archive/refs/heads/main.zip) をダウンロードし、展開します。
3. 展開先の **macOS：`Start-macOS.command`／Windows：`Start-Windows.bat`** を開きます。Linux、またはファイルから起動できない場合は、展開先で `python3 launch.py` を実行します。
4. 初回のダウンロードが終わるとブラウザーが開きます。表示されたURLをChrome / Edgeで開き、モデルを読み込んで内蔵デモを試します。

起動したウィンドウは開いたまま使い、終了するにはそこで **`Ctrl+C`** を押します。

初回は約323MBを追加ダウンロードし、約400MBを展開します。一時ファイルを含め、空き容量は1GB程度を用意してください。ランチャーが取得・SHA-256照合・ローカルサーバーの起動を行うため、GitやPyTorchなどの追加パッケージは不要です。

ダウンロードが途切れた場合は自動で再試行します。終了してしまっても、同じフォルダーで再度起動すると保存済みの途中データから再開します。配信先が途中再開に対応しない場合は最初から取得します。

GitHubのページを開くだけで音を試すことはできません。このリポジトリーは、ローカルで動かすためのコードと追加資産を配布しており、公開Webサービスではありません。

## この実装について

入力した音声を、学習済みモデルの encoder → 潜在表現 → decoder に通します。音声処理はブラウザー内で実行し、付属サーバーは静的ファイルだけを配信します。外部の音声処理 API は使用しません。

**Neutone Morpho とは独立した実装です。Morpho のエンジン・モデル・バイナリーは含みません。** RAVE と各モデルの作者・提供元による公式製品や推奨実装でもありません。

## 利用できる人と、必要な準備

非商用の条件と出典表示を守る方は、このコードを取得・実行・改変できます。作者本人だけに利用を限定しているわけではありません。商用利用には、該当するコード・モデルの権利者から別途許諾が必要です。

- **必要環境**：Python 3.9以上を入れたPCと、Chrome / Edgeなどの対応ブラウザー。Pythonは [公式サイト](https://www.python.org/downloads/) から取得できます。
- **確認範囲**：macOS上のChromium系ブラウザーでの検証記録があります。Windows / Linuxは同等の実機検証をしておらず、すべてのPCでの動作を保証するものではありません。
- **音声の安定性**：実マイクでは瞬間的な音切れが残ります。参加者へそのまま配る完成アプリではなく、試しながら使う研究実装です。

## English overview

A personal, noncommercial research prototype for running pretrained RAVE audio-to-audio models in the browser. It combines AudioWorklet, a worker, and ONNX Runtime Web to explore stateful inference and latent controls. Audio processing runs locally; this repository publishes source and downloadable research assets, not a hosted service. It contains no Neutone Morpho engine or model assets. Live latency and uninterrupted playback are not guaranteed.

## Gitから取得して起動する

必要なものは Python 3.9 以上と、AudioWorklet / WebAssembly に対応したブラウザーです。まず Chromium 系ブラウザーで試してください。WebGPU は対応環境で選択できる実験的な経路です。

```sh
git clone https://github.com/khrtkn/rave-web.git
cd rave-web
python3 launch.py
```

起動時に表示されたURLをブラウザーで開きます。通常は [http://127.0.0.1:8873](http://127.0.0.1:8873) です。使用中の場合は空いているポートを選びます。

- ランチャーから呼ばれる `fetch-assets.py` は [v0.1.0 の Release assets](https://github.com/khrtkn/rave-web/releases/tag/v0.1.0) からモデルと実行環境を復元します。Git のソースだけでは推論できません。
- 通常の起動に PyTorch や ONNX の Python パッケージは不要です。
- `file://` で HTML を直接開かず、付属サーバーを使用してください。サーバーは `127.0.0.1` にのみ接続を受け付け、SharedArrayBuffer に必要な分離ヘッダーも設定します。
- 比較用の旧版は [http://127.0.0.1:8873/legacy/](http://127.0.0.1:8873/legacy/)、ベンチマーク画面は [http://127.0.0.1:8873/benchmark.html](http://127.0.0.1:8873/benchmark.html) です。

### 音を試す

1. モデルを選び、読み込みが完了するまで待ちます。既定は Organ 512 です。
2. 出力音量を小さくし、まず内蔵デモ入力で動作を確認します。
3. 必要に応じてローカルの音声ファイル、またはマイク・オーディオインターフェースへ入力を切り替えます。マイク使用時はブラウザーの許可が必要です。
4. 潜在座標の gain / bias、Dry / Wet、出力音量を操作します。
5. 終了時は停止操作を行います。マイクのトラックと AudioContext を解放します。

モデルの生出力は大きな値や初期化時の過渡音を含むことがあります。低い音量から試してください。主画面の再生経路には出力ピークの制限がありますが、数値比較用の生出力や録音データと同じ信号ではありません。

## モデル

すべて 48 kHz・モノラルの学習済みモデルです。本プロジェクトでは再学習せず、ONNX 化、過去状態の入出力化、潜在座標操作などを加えています。

| モデル | 1 回の処理サンプル数 | その音声の長さ | 公開する操作座標 | 位置づけ |
| --- | ---: | ---: | --- | --- |
| Organ 512 | 512 | 10.67 ms | 保存済み PCA の 6 方向 | 主な研究対象・既定 |
| Organ 1024 | 1,024 | 21.33 ms | 同じ PCA の 6 方向 | 同じ重みで 512 ステップを 2 回処理する比較版 |
| Water 2048 | 2,048 | 42.67 ms | 保持された 16 座標のうち 6 座標 | Water / Pond Brain の比較版 |
| Water 4096 | 4,096 | 85.33 ms | 同上 | 処理単位の比較版 |
| Djembe 2048 | 2,048 | 42.67 ms | 保持された 4 座標 | 小型モデルの実験的な比較版 |

**表の ms は処理する音声区間の長さです。マイクからスピーカーまでの遅延ではありません。** Water の座標を PCA と呼ぶ根拠はなく、Djembe の投影も PCA 由来とは確認できていません。いずれの座標にも、感情や音の意味を表す名前は割り当てていません。

出典・固定リビジョン・変更内容は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) と、各モデルディレクトリーの attribution / contract を参照してください。モデルのファイル名、サイズ、SHA-256 はモデルカタログと配布資産の manifest で管理します。

## 実装の概要

```text
デモ / 音声ファイル / マイク
    → Web Audio / AudioWorklet
    → Worker 内の ONNX Runtime Web
    → 状態を引き継ぐ RAVE encoder / decoder
    → 出力音量・ピーク制限
    → ブラウザーの音声出力
```

- WASM と、対応環境での WebGPU 経路があります。
- 分離されたページでは SharedArrayBuffer と Atomics のリングバッファーを使い、利用できない場合はメッセージによる経路に切り替えます。
- 過去状態を次の推論へ引き継ぎます。入力ブロックごとに独立した変換を繰り返す方式ではありません。
- 演奏優先設定では、再生中の連続的な波形・数値描画を減らします。音声処理や操作は継続します。
- 音声ファイルとマイク入力はブラウザー内で扱います。付属サーバーに音声アップロード用の API はありません。

## 構成

```text
launch.py                        追加資産の取得・ローカル起動・ブラウザー表示
Start-macOS.command              macOS用ランチャー
Start-Windows.bat                Windows用ランチャー
serve.py                         ローカル起動用の入口
scripts/fetch-assets.py           Release assets の復元
outputs/
  rave-morph-lowlatency/
    web/                         主なブラウザー実装
    model-tools/                 Organ の取得・変換・検証スクリプト
    tests/                       Node.js の単体テスト
  rave-morph/
    web/                         比較用の旧版
    model-tools/                 Water / Djembe の変換・検証スクリプト
    tests/                       旧版の単体テスト
```

履歴との対応とビルドスクリプトの再現性のため、`outputs/` 以下の元のディレクトリー名を維持しています。`rave-morph` はこの実験のディレクトリー名であり、Neutone Morpho の実装を示すものではありません。

モデルと ONNX Runtime の大きなバイナリーは Git に含めず、Release assets から復元します。個人のマイク収録、過去の収録音声、端末・入力機器などの個人情報を含む検証ログは配布対象に含めません。

## テストとモデルの再構築

Node.js がある場合、リポジトリーのルートで単体テストを実行できます。

```sh
node --test outputs/rave-morph-lowlatency/tests/*.test.mjs
python3 -m unittest discover -s scripts/tests -v
```

これらのテストは音声処理の補助関数、ストリームの状態管理、ダウンロードの中断・再開を確認します。実機の入出力遅延、ブラウザーの継続再生、聴感上の品質を証明するものではありません。ブラウザーではモデルの読み込み、内蔵デモ、停止、数値検証をそれぞれ確認してください。

モデルを元データから再構築する場合は、[Organ の model-tools](outputs/rave-morph-lowlatency/model-tools/README.md) と [Water / Djembe の model-tools](outputs/rave-morph/model-tools/README.md) を参照してください。こちらは追加の Python 依存関係と大きな取得ファイルを必要とします。Water の再構築は、明記された中間 ONNX を入力とする手順であり、上流 TorchScript からの変換経路を再現したものではありません。

## 制限と確認できていないこと

- **ライブ用途の遅延・連続再生を保証しません。** 実験中には停止・欠落が発生しています。画面内の処理時間や経路の目標値を、物理的な入出力の実測値として扱わないでください。
- ブラウザー、CPU / GPU、オーディオ機器、サンプルレート変換などで挙動が変わります。すべての環境での互換性を確認していません。
- モデルは入力音声を再構成します。言葉の意味、話者、音高、リズムを必ず保つ変換ではありません。
- Water は元の中間 ONNX に対して検証しています。上流 TorchScript との同一性は確認していません。
- Djembe は既知の追加シフト 1,360 サンプル（約 28.33 ms）を伴う実験版です。元モデル自体の絶対的な入出力のずれは未確認で、標準正規ノイズを与えた比較には数値的な残差があります。
- 数値比較の一致は、聴感上の良さや他製品との同等性を意味しません。

## ライセンス

本プロジェクトの独自の制作物は [CC BY-NC 4.0](LICENSE) で公開します。**非商用利用を条件とするソース公開です。商用利用を許可するライセンスではありません。** 作者・出典・ライセンス表示を保持し、変更した場合はその旨を示してください。

モデルにも CC BY-NC 4.0 が適用されます。ONNX に変換したことや GitHub でソースを公開したことによって、モデルの非商用条件がなくなるわけではありません。営利利用には、該当する権利者から別途許諾を得る必要があります。

ONNX Runtime Web は Microsoft とその貢献者による **MIT License** の別作品です。付属の `LICENSE` と `ThirdPartyNotices.txt` を保持してください。本リポジトリーの CC BY-NC 表示で、これらの第三者ライセンスを変更するものではありません。

詳しい帰属表示・ライセンスの適用範囲は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)、CC の正式な条件は [CC BY-NC 4.0 Legal Code](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en) を参照してください。
