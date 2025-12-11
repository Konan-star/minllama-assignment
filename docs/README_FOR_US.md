# Min-Llama

## 準備

### 環境構築
- Python 3.12 (`.python-version`)
- `pyproject.toml` + uv による依存関係管理

### モデルダウンロード
```bash
curl -O https://www.cs.cmu.edu/~vijayv/stories42M.pt
```

---

## 実装箇所 (#todoマーク)

### rope.py
Rotary Positional Embeddings (RoPE)

| 関数 | 内容 |
|------|------|
| `apply_rotary_emb()` | 回転行列による位置エンコーディング |

テスト: `python rope_test.py`

### llama.py
Llama2モデルコア

| 関数 | 内容 | 使用タスク |
|------|------|-----------|
| `RMSNorm.norm()` | Root Mean Square正規化 | 全タスク |
| `Attention.forward()` | マルチヘッドアテンション | 全タスク |
| `Llama.forward()` | モデル全体のforward pass | 全タスク |
| `Llama.generate()` | Temperature samplingテキスト生成 | タスク1 |

テスト: `python sanity_check.py`

### optimizer.py
AdamWオプティマイザー

| 関数 | 内容 | 使用タスク |
|------|------|-----------|
| `AdamW.step()` | パラメータ更新 | タスク3 |

テスト: `python optimizer_test.py`

### classifier.py
分類ヘッド

| 関数 | 内容 | 使用タスク |
|------|------|-----------|
| `LlamaEmbeddingClassifier.forward()` | 隠れ状態 → dropout → 線形層 | タスク3 |

注: `LlamaZeroShotClassifier`は実装済み

---

## タスク詳細

### タスク1: テキスト生成 (generate)

```bash
python run_llama.py --option generate
```

プロンプト: "I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is"

Temperature:
- 0.0: 決定的生成（最確率トークン選択）
- 1.0: 確率的生成（ランダムサンプリング）

コードフロー (`generate_sentence`):
```python
llama = load_pretrained(args.pretrained_model_path)
start_ids = enc.encode(prefix, bos=True, eos=False)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# [実装] Llama.generate()
y = llama.generate(x, max_new_tokens, temperature=temperature)

sentence = enc.decode(y[0].tolist())
```

出力:
- `generated-sentence-temp-0.txt`
- `generated-sentence-temp-1.txt`



### タスク2: ゼロショット分類 (prompt)

パラメータ更新なしで分類

コマンド (SST):
```bash
python run_llama.py --option prompt --batch_size 10 \
  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt \
  --label-names data/sst-label-mapping.json \
  --dev_out sst-dev-prompting-output.txt --test_out sst-test-prompting-output.txt
```

コマンド (CFIMDB):
```bash
python run_llama.py --option prompt --batch_size 10 \
  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt \
  --label-names data/cfimdb-label-mapping.json \
  --dev_out cfimdb-dev-prompting-output.txt --test_out cfimdb-test-prompting-output.txt
```

コードフロー (`test_with_prompting`):
```python
label_names = json.load(open(args.label_names, 'r'))
# SST: ["awful", "bad", "average", "good", "excellent"]
# CFIMDB: ["negative", "positive"]

prompt_suffix = f"Is this movie {label_name_str}? This movie is "

model = LlamaZeroShotClassifier(config, tokenizer, label_names)
# requires_grad = False

dev_data = create_data(args.dev, tokenizer, 'valid', 
                      eos=False, prompt_suffix=prompt_suffix)
# 例: "Great movie" → "Great movie Is this movie awful, bad, average, good, or excellent? This movie is "

# 各ラベル単語の次トークン確率を計算して最大のものを選択
dev_acc = model_eval(dev_dataloader, model, device)
```

仕組み: 入力文+プロンプト → 各ラベル単語の出現確率 → 最大確率のラベルを選択

結果:
- SST: ~0.21 (ランダムレベル)
- CFIMDB: ~0.50 (ランダムレベル)

出力:
- `sst-dev-prompting-output.txt`, `sst-test-prompting-output.txt`
- `cfimdb-dev-prompting-output.txt`, `cfimdb-test-prompting-output.txt`

注: 実装不要 (`LlamaZeroShotClassifier`は実装済み)


### タスク3: ファインチューニング (finetune)

分類タスク特化の学習

コマンド (SST):
```bash
python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 80 \
  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt \
  --label-names data/sst-label-mapping.json \
  --dev_out sst-dev-finetuning-output.txt --test_out sst-test-finetuning-output.txt
```

コマンド (CFIMDB):
```bash
python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 10 \
  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt \
  --label-names data/cfimdb-label-mapping.json \
  --dev_out cfimdb-dev-finetuning-output.txt --test_out cfimdb-test-finetuning-output.txt
```

コードフロー:

学習 (`train`):
```python
train_data, num_labels = create_data(args.train, tokenizer, 'train')
dev_data = create_data(args.dev, tokenizer, 'valid')

model = LlamaEmbeddingClassifier(config)
# requires_grad = True (Llamaモデル全体+分類ヘッド)

# [実装] optimizer.AdamW
optimizer = AdamW(model.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    for batch in train_dataloader:
        # [実装] LlamaEmbeddingClassifier.forward()
        logits = model(b_ids)
        
        loss = F.nll_loss(logits, b_labels)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
    
    dev_acc = model_eval(dev_dataloader, model, device)
    if dev_acc > best_dev_acc:
        save_model(model, optimizer, args, config, args.filepath)
```

評価 (`test`):
```python
saved = torch.load(args.filepath)
model = LlamaEmbeddingClassifier(config)
model.load_state_dict(saved['model'])

dev_acc = model_eval(dev_dataloader, model, device)
test_acc = model_eval(test_dataloader, model, device)

write_predictions_to_file("dev", args.dev_out, dev_acc, dev_pred, dev_sents)
write_predictions_to_file("test", args.test_out, test_acc, test_pred, test_sents)
```

結果:
- SST: ~0.41 (タスク2: 0.21から改善)
- CFIMDB: ~0.80 (タスク2: 0.50から改善)

出力:
- `sst-dev-finetuning-output.txt`, `sst-test-finetuning-output.txt`
- `cfimdb-dev-finetuning-output.txt`, `cfimdb-test-finetuning-output.txt`

実装必要:
- `classifier.LlamaEmbeddingClassifier.forward()`
- `optimizer.AdamW.step()`

---

## データセット

| データセット | クラス | ラベル | ファイル |
|------------|-------|--------|---------|
| SST | 5 | ["awful", "bad", "average", "good", "excellent"] | `data/sst-*.txt` |
| CFIMDB | 2 | ["negative", "positive"] | `data/cfimdb-*.txt` |

---

## テスト

| ファイル | 対象 | コマンド |
|---------|------|----------|
| `rope_test.py` | RoPE | `python rope_test.py` |
| `optimizer_test.py` | AdamW | `python optimizer_test.py` |
| `sanity_check.py` | Llama forward pass | `python sanity_check.py` |
