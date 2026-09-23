import torch

from test_independent_latent_sid import f, model


SCALE = 1.2593601909739016


def test_same_forward_and_only_gps_reconstruction_gradient_scales():
    a, b = model(), model()
    b.reconstruction_scale = SCALE
    a.train(); b.train()
    batch = f.complete_batch()
    torch.manual_seed(23)
    original = a(batch)
    torch.manual_seed(23)
    scaled = b(batch)
    for key in ("codes", "loss", "reconstruction_loss", "quant_loss", "decoder_latent"):
        torch.testing.assert_close(original[key], scaled[key], rtol=0, atol=0)
    original["reconstruction_loss"].backward()
    scaled["reconstruction_loss"].backward()
    for view in ("grid", "road"):
        assert all(p.grad is None or p.grad.abs().sum() == 0 for p in b.view_encoders[view].parameters())
    for pa, pb in zip(a.view_projections["gps"].parameters(), b.view_projections["gps"].parameters()):
        torch.testing.assert_close(pb.grad, pa.grad * SCALE, rtol=1e-5, atol=1e-6)
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in b.rq_quant.parameters())


def test_quantizer_loss_gradient_does_not_change():
    a, b = model(), model()
    b.reconstruction_scale = SCALE
    d = a.rq_quant.codebooks[0].codebook.embedding_dim
    latents = {v: torch.randn(5, d, requires_grad=True) for v in ("grid", "road", "gps")}
    clones = {v: z.detach().clone().requires_grad_() for v, z in latents.items()}
    a.conditional_quantize(latents).quant_loss.backward()
    b.conditional_quantize(clones).quant_loss.backward()
    for view in latents:
        torch.testing.assert_close(latents[view].grad, clones[view].grad, rtol=0, atol=0)
