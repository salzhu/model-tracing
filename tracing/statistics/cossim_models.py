import torch

def statistic(base_model, ft_model):
    return calculate_cos_sim(base_model, ft_model)

def calculate_cos_sim(model1, model2):
    total_cos_sim = 0
    num_layers = 0

    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        if name1 != name2:
            raise ValueError(f"Model parameter names do not match: {name1} != {name2}")
        elif param1.shape != param2.shape:
            if name1 == "model.embed_tokens.weight" or name1 == "lm_head.weight":
                # print(
                #     f"Skipping {name1} because of shape mismatch: {param1.shape} != {param2.shape}"
                # )
                continue
            raise ValueError(
                f"Model parameter shapes do not match for {name1}: {param1.shape} != {param2.shape}"
            )

        param1 = torch.flatten(param1)
        param2 = torch.flatten(param2)

        cos_sim = torch.dot(param1, param2) / (torch.linalg.norm(param1) * torch.linalg.norm(param2))
        total_cos_sim += cos_sim.item()
        num_layers += 1

    avg_cos_sim = total_cos_sim / num_layers
    return avg_cos_sim