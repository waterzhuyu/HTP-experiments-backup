import torch
from independent_latent_sid.gradient_conflict import cosine

def test_gradient_direction_and_missing():
    x=torch.tensor([1.,2.]);y=torch.tensor([-2.,1.])
    assert abs(cosine(x,-x)+1)<1e-6
    assert abs(cosine(x,y))<1e-6
    assert cosine(x,None) is None
    assert cosine(x,torch.zeros(2)) is None
