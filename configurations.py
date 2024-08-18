import os
import git

def generate_config(dataset, args, exp_name):
    repo = git.Repo(search_parent_directories=True)

    config = {
        'model': {
            'encoder': {
                'type': args.encoder,
            },
            'decoder': {
                'attention': {
                    'type': 'tanh'
                },
                'output_size': dataset.output_size
            }
        },
        'training': {
            'bsize': args.bsize,
            'weight_decay': args.weight_decay,
            'pos_weight': dataset.pos_weight if hasattr(dataset, 'pos_weight') else None,
            'basepath': dataset.basepath if hasattr(dataset, 'basepath') else 'outputs',
            'exp_dirname': os.path.join(dataset.name, exp_name)
        },
        'git_info': {
            'branch': repo.active_branch.name,
            'sha': repo.head.object.hexsha
        },
        'command': args.command
    }

    if args.encoder == 'average':
        config['model']['encoder'].update({'projection': True, 'activation': 'tanh'})

    if args.encoder != 'bert':
        config['model']['encoder'].update({'vocab_size': dataset.vocab_size,
        'embed_size': dataset.word_dim,
        'hidden_size': args.hidden_size,})

    return config