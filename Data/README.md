---

Files:

This directory is empty by default, other than this README.md file.
Upon running the `make data` for the top level directory, the data will be downloaded using 'gdown' and decompressed with the following dictory structure.

.
├── README.md
├── sp24_grad_project_data
│   ├── nlp-arena-data
│   │   ├── arena-validation-set-prompt-only.jsonl.gz
│   │   ├── chatbot-arena-conversations.jsonl.gz
│   │   ├── chatbot-arena-gpt3-scores.jsonl.gz
│   │   └── chatbot-arena-prompts-embeddings.npy
│   └── satellite-image-data
│       ├── hurricane-matthew
│       │   ├── train_images.npz
│       │   └── train_labels.npy
│       ├── midwest-flooding
│       │   ├── train_images.npz
│       │   └── train_labels.npy
│       ├── socal-fire
│       │   ├── train_images.npz
│       │   └── train_labels.npy
│       ├── test_images_flooding-fire.npz
│       └── test_images_hurricane-matthew.npz
└── sp24_grad_project_data.zip
