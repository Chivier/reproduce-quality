import spacy
from utils import encrypte_noun_text
import os
import json

# 加载预训练的英语模型
nlp = spacy.load("en_core_web_trf")

# 输入文本
text = "The quick brown fox jumps over the lazy dog."
text = """Affective events BIBREF0 are events that typically affect people in positive or negative ways. For example, getting money and playing sports are usually positive to the experiencers; catching cold and losing one's wallet are negative. Understanding affective events is important to various natural language processing (NLP) applications such as dialogue systems BIBREF1, question-answering systems BIBREF2, and humor recognition BIBREF3. In this paper, we work on recognizing the polarity of an affective event that is represented by a score ranging from $-1$ (negative) to 1 (positive).
Learning affective events is challenging because, as the examples above suggest, the polarity of an event is not necessarily predictable from its constituent words. Combined with the unbounded combinatorial nature of language, the non-compositionality of affective polarity entails the need for large amounts of world knowledge, which can hardly be learned from small annotated data.
In this paper, we propose a simple and effective method for learning affective events that only requires a very small seed lexicon and a large raw corpus. As illustrated in Figure FIGREF1, our key idea is that we can exploit discourse relations BIBREF4 to efficiently propagate polarity from seed predicates that directly report one's emotions (e.g., “to be glad” is positive). Suppose that events $x_1$ are $x_2$ are in the discourse relation of Cause (i.e., $x_1$ causes $x_2$). If the seed lexicon suggests $x_2$ is positive, $x_1$ is also likely to be positive because it triggers the positive emotion. The fact that $x_2$ is known to be negative indicates the negative polarity of $x_1$. Similarly, if $x_1$ and $x_2$ are in the discourse relation of Concession (i.e., $x_2$ in spite of $x_1$), the reverse of $x_2$'s polarity can be propagated to $x_1$. Even if $x_2$'s polarity is not known in advance, we can exploit the tendency of $x_1$ and $x_2$ to be of the same polarity (for Cause) or of the reverse polarity (for Concession) although the heuristic is not exempt from counterexamples. We transform this idea into objective functions and train neural network models that predict the polarity of a given event.
We trained the models using a Japanese web corpus. Given the minimum amount of supervision, they performed well. In addition, the combination of annotated and unannotated data yielded a gain over a purely supervised baseline when labeled data were small.
Learning affective events is closely related to sentiment analysis. Whereas sentiment analysis usually focuses on the polarity of what are described (e.g., movies), we work on how people are typically affected by events. In sentiment analysis, much attention has been paid to compositionality. Word-level polarity BIBREF5, BIBREF6, BIBREF7 and the roles of negation and intensification BIBREF8, BIBREF6, BIBREF9 are among the most important topics. In contrast, we are more interested in recognizing the sentiment polarity of an event that pertains to commonsense knowledge (e.g., getting money and catching cold).
Label propagation from seed instances is a common approach to inducing sentiment polarities. While BIBREF5 and BIBREF10 worked on word- and phrase-level polarities, BIBREF0 dealt with event-level polarities. BIBREF5 and BIBREF10 linked instances using co-occurrence information and/or phrase-level coordinations (e.g., “$A$ and $B$” and “$A$ but $B$”). We shift our scope to event pairs that are more complex than phrase pairs, and consequently exploit discourse connectives as event-level counterparts of phrase-level conjunctions.
BIBREF0 constructed a network of events using word embedding-derived similarities. Compared with this method, our discourse relation-based linking of events is much simpler and more intuitive."""

# 使用模型处理文本
doc = nlp(text)


folder = './parsed_data/quality_v1.0.1'

def hide_text(raw_input, spacy_model, uuid_map):
    return encrypte_noun_text(raw_input, spacy_model, uuid_map)

uuid_file = os.path.join(folder, 'encrypt_1.json')

# Load or initialize UUID mapping
if os.path.exists(uuid_file):
    with open(uuid_file, 'r', encoding='utf-8') as f:
        uuid_map = json.load(f)
else:
    uuid_map = {}

print(hide_text(text, nlp, uuid_map))
