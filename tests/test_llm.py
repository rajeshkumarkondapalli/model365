import torch

from simple_llm.model import GPT
from simple_llm.tokenizer import CharTokenizer
from simple_llm.train import get_batch, load_corpus, train


def test_tokenizer_roundtrip():
    tokenizer = CharTokenizer("hello world")
    ids = tokenizer.encode("hello")
    assert tokenizer.decode(ids) == "hello"


def test_forward_pass_shapes():
    tokenizer = CharTokenizer("abcdefgh")
    model = GPT(vocab_size=tokenizer.vocab_size, block_size=8, n_embd=16, n_head=2, n_layer=2)
    idx = torch.randint(0, tokenizer.vocab_size, (4, 8))
    targets = torch.randint(0, tokenizer.vocab_size, (4, 8))

    logits, loss = model(idx, targets)

    assert logits.shape == (4, 8, tokenizer.vocab_size)
    assert loss.dim() == 0


def test_generate_produces_requested_length():
    tokenizer = CharTokenizer("abcdefgh")
    model = GPT(vocab_size=tokenizer.vocab_size, block_size=8, n_embd=16, n_head=2, n_layer=2)
    context = torch.zeros((1, 1), dtype=torch.long)

    generated = model.generate(context, max_new_tokens=20)

    assert generated.shape == (1, 21)


def test_loss_decreases_during_training():
    torch.manual_seed(0)
    text = load_corpus(repeats=20)
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    generator = torch.Generator().manual_seed(0)

    model = GPT(vocab_size=tokenizer.vocab_size, block_size=32, n_embd=32, n_head=4, n_layer=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    xb, yb = get_batch(data, 32, 16, generator)
    _, first_loss = model(xb, yb)

    for _ in range(100):
        xb, yb = get_batch(data, 32, 16, generator)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    assert loss.item() < first_loss.item()


def test_end_to_end_training_learns_corpus():
    model, tokenizer = train(steps=500)
    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, max_new_tokens=50)[0].tolist()
    text = tokenizer.decode(generated)

    assert len(text) == 51
    assert all(ch in tokenizer.stoi for ch in text)
