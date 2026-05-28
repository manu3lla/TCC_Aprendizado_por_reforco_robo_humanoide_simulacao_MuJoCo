import torch.nn as nn


def _nome_classe(objeto):
    if objeto is None:
        return "None"

    if isinstance(objeto, type):
        return objeto.__name__

    return objeto.__class__.__name__


def _formatar_espaco(nome, espaco):
    linhas = [f"{nome}: {espaco}"]

    shape = getattr(espaco, "shape", None)
    if shape is not None:
        linhas.append(f"{nome} shape: {shape}")

    n = getattr(espaco, "n", None)
    if n is not None:
        linhas.append(f"{nome} n: {n}")

    return linhas


def formatar_especificacoes_rede(model, env=None):
    policy = model.policy

    observation_space = getattr(model, "observation_space", None)
    action_space = getattr(model, "action_space", None)

    if env is not None:
        observation_space = getattr(env, "observation_space", observation_space)
        action_space = getattr(env, "action_space", action_space)

    linhas = [
        "",
        "=" * 60,
        "ESPECIFICACOES DA REDE NEURAL",
        "=" * 60,
        "",
        "Espacos do ambiente/modelo",
        "-" * 60,
    ]

    linhas.extend(_formatar_espaco("Observation space", observation_space))
    linhas.extend(_formatar_espaco("Action space", action_space))

    linhas.extend(
        [
            "",
            "Politica",
            "-" * 60,
            f"Classe da policy: {_nome_classe(policy)}",
            f"Arquitetura net_arch: {getattr(policy, 'net_arch', None)}",
            f"Funcao de ativacao: {_nome_classe(getattr(policy, 'activation_fn', None))}",
            f"Extrator de features: {_nome_classe(getattr(policy, 'features_extractor', None))}",
            "",
            "MLP extractor",
            "-" * 60,
            str(getattr(policy, "mlp_extractor", None)),
            "",
            "Camada final da politica (actor)",
            "-" * 60,
            str(getattr(policy, "action_net", None)),
            "",
            "Camada final do valor (critic)",
            "-" * 60,
            str(getattr(policy, "value_net", None)),
            "",
            "Camadas lineares",
            "-" * 60,
        ]
    )

    for nome, modulo in policy.named_modules():
        if isinstance(modulo, nn.Linear):
            linhas.append(f"{nome}: {modulo.in_features} -> {modulo.out_features}")

    total_params = sum(param.numel() for param in policy.parameters())
    trainable_params = sum(
        param.numel() for param in policy.parameters() if param.requires_grad
    )

    linhas.extend(
        [
            "",
            "Parametros",
            "-" * 60,
            f"Total de parametros: {total_params}",
            f"Parametros treinaveis: {trainable_params}",
            "=" * 60,
            "",
        ]
    )

    return "\n".join(linhas)


def imprimir_especificacoes_rede(model, env=None, output_path=None):
    texto = formatar_especificacoes_rede(model, env=env)
    print(texto)

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as arquivo:
            arquivo.write(texto)
            arquivo.write("\n")

        print(f"Especificacoes da rede salvas em: {output_path}")
