# CookedGPT
This project is General Pretrained Transformer (GPT) which utilizes data grabbed from a Discord server. The frontend is built with Angular and the model responses and queries are served with Flask.

### Purpose
I wanted to learn how ChatGPT works. Recently, I was following through MIT's Artificial Intelligence (6.034) lectures and Patrick Winston said that **meta-knowledge, knowledge about knowledge, gives us power** as we can give names to elusive concepts or things. I wanted some of that power. I didn't know how ChatGPT worked, of course other than on a higher level, so I wanted to actually comprehend it given how prevelant it is in modern life. I found Andrej's video about building a GPT so helpful but at the same time made me realize I know less than when I started. Overall, I wanted to know how a language model worked, now I'm left with a lot more questions.

### Thanks
- [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) allowed me to grab messages from a server, then clean it with `clean.py` to feed to the model
- [Andrej Karpathy](https://youtu.be/kCc8FmEb1nY?si=WB1CDbZFmxV5kxYY) explained how to create a simple GPT
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) was supplementary to Andrej's video which allowed me to more deeply understand

### TODO
- Tenor gifs; look up using last part of URL?
- Interface w/ Angular
- Dockerize