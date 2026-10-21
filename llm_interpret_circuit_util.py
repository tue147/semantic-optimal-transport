import json
from bs4 import BeautifulSoup
from typing import Any, Literal, TypeAlias

Messages: TypeAlias = list[dict[Literal["role", "content"], str]]

def parse_logits_html(html_str):
    """
    Req #4: Parses the logits HTML to clearly separate Top, Bottom, and Output tokens.
    """
    if not html_str or html_str in ["N/A", "Error"]:
        return "No data"
    
    soup = BeautifulSoup(html_str, "html.parser")
    sections = []
    current_header = None
    current_tokens = []
    
    main_div = soup.find("div")
    if not main_div: 
        return "No data"
    
    for child in main_div.children:
        if child.name == "div": # type: ignore
            # This indicates a header like "Top", "Bottom", or "Output Tokens"
            if current_header and current_tokens:
                sections.append(f"{current_header}: [" + ", ".join(current_tokens) + "]")
            current_header = child.get_text(strip=True)
            current_tokens = []
        elif child.name == "span": # type: ignore
            # This is the pill containing "token : value"
            current_tokens.append(child.get_text(strip=True))
            
    # Catch the last section
    if current_header and current_tokens:
        sections.append(f"{current_header}: [" + ", ".join(current_tokens) + "]")
        
    return " | ".join(sections)

def parse_act_html(html_str, include_full_context=False):
    """
    Req #1 & #2: Parses the activation HTML to create structured text.
    Toggle conditional: include_full_context controls whether the full string is added.
    """
    if not html_str or "No activation data" in html_str:
        return "No data"
    
    soup = BeautifulSoup(html_str, "html.parser")
    out = []
    
    # Extract Global Max Act
    header = soup.find(class_="act-header")
    if header:
        out.append(f"    {header.get_text(strip=True)}")
        
    for block in soup.find_all(class_="interval-block"):
        # Req #1: Add a clear interval header
        int_header = block.find(class_="interval-header")
        if int_header:
            spans = int_header.find_all("span")
            if len(spans) == 2:
                out.append(f"    --- Interval: [{spans[0].get_text()}], Density: {spans[1].get_text()} ---")
        
        # Req #2: Parse examples nicely
        for row in block.find_all(class_="example-row"):
            act_val_tag = row.find(class_="act-val")
            act_val = act_val_tag.get_text(strip=True) if act_val_tag else "0.00"
            
            def extract_tokens(view_div):
                if not view_div: return ""
                toks = []
                for span in view_div.find_all("span", class_="token"):
                    is_target = "target-token" in span.get("class", [])
                    
                    tooltip = span.find(class_="tooltip")
                    val = tooltip.get_text(strip=True) if tooltip else "0"
                    
                    # Extract token text, stripping out the tooltip text that get_text() merges
                    tok_text = span.get_text(strip=True)
                    if tooltip and tok_text.endswith(val):
                        tok_text = tok_text[:-len(val)]
                        
                    formatted_tok = f"{tok_text}({val})"
                    if is_target:
                        formatted_tok = f"**{formatted_tok}**" # Highlight the target token
                    toks.append(formatted_tok)
                return " ".join(toks)

            short_view = row.find(class_="short-view")
            out.append(f"      > Target Act: {act_val}")
            
            if short_view:
                out.append(f"        Short Context: {extract_tokens(short_view)}")
                
            # Toggle parameter for full string examples
            if include_full_context:
                full_view = row.find(class_="full-view")
                if full_view:
                    out.append(f"        Full Context:  {extract_tokens(full_view)}")
                    
    return "\n".join(out)

prompt = '''
You are an expert AI interpretability researcher analyzing the internal computational graph of a transformer language model. 

I am providing you with a causal graph made of Sparse Autoencoder (SAE) features. The graph represents how the model processes a specific sequence of tokens to make a prediction for the next token.

Here are your strict interpretation rules:
1. THE GOLD STANDARD IS ACTIVATION PATTERNS: The most important way to interpret a feature is by looking at its "Act Context" (the tokens that activate it). Look for similar pattern in the activation examples (i.e. similar contexts, tokens, ...; sometimes the pattern can be subtle such as "plural noun" or "tracking the third token of the input", ...). 
2. LOGIT INFORMATION: Look at the "Logits" data which show the tokens the feature promotes/suppresses. If the promoted/suppressed tokens have similar meaning or patern, it show that the feature promotes certain concept. If the promoted/suppressed tokens seem entirely random and disconnected from the activation context, disregard the logit information.
3. CAUSALITY: Review the "EDGES" section. Connections between features (or clusters) with higher weights/scores are causally more important to the model's final output. Pay attention to the flow from earlier layers/sequence positions to later ones.

DATA IN THE GRAPH:
- Nodes: Can be individual SAE features or "Clusters" of features, which is paired with a importance score.
- Feature: Uniquely identified by tuple (hook layer, sequence position, id), contains information about activation contexts, tokens that the feature promotes/supresses, and how it contributes to the output tokens.
- Act Context: Shows text intervals and target values where the feature fired.
- Logits: Shows the top promoted tokens, top suppressed tokens, and specific output tokens affected.
- Edges: Represent causal connections between nodes.

EXAMPLE OF FEATURE INFORMATION:
- Feature: (Hook And Layer)::(Sequence Position)::feat::(id) | (layer) | (sequence position) - (input token w.r.t to sequence position)
    Causality Score: (importance score)
    Act Context:
        Max Act:(max activation)
        --- Interval: [(activation range)], Density: (activation frequency of the interval)% ---
        > Target Act: (activation of the target token)
        Short Context: (token)(activation) ... **(target token)(activation)**...
        Full Context (OPTIONAL): ...
        ...
    Logits (Promotes/Suppresses):
    Top: [(token) : (value), ...] | Bottom: [(token) : (value), ...] | Output Tokens: [(token) : (value), ...]

TASK:
Based on the provided graph data, complete the following:

1. Review and caption the meaning of each feature/cluster with 1-2 SHORT sentences. The caption should be general that reflects the meaning of the feature/cluster of features, don't be overly specific.
2. Review the edge connections between the features/clusters. Summarize which feature/cluster is causally important to which.
3. Summarize the steps of how the model processes the input into exactly 5 bullet points. Focus on the progression of information through the sequence and layers, highlighting the most causally important features/clusters.
4. Finally, based solely on those 5 steps, write exactly one or two sentences concluding how the model performs the computation from the initial input to the final output.

OUTPUT FORMAT:
### STEP 1:
feature / cluster (id): (caption)
...
### STEP 2:
feature / cluster (id): is important for features / clusters ...
...
### STEP 3:
1, (describe how information is processed and how model outputs)
2, ...
### STEP 4:
Summary: (your summarization) 

IMPORTANT NOTE:
- In step 3 and 4, do NOT show any clue about you are interpreting using cluster of feature or individual feature (do not say "cluster (id) / feature (id) shows that ...", instead "the model recognizes that ...")
- Logit information can be a good source to interpret features, if the top / bottom tokens represent similar meaning.
'''

'''
LLM circuit clustering
'''

def format_features_for_llm_clustering(graph_data, include_full_context: bool = False) -> str:
    """
    Extracts ONLY feature nodes from the graph data and formats them 
    for the LLM to read, completely ignoring edges.
    """
    output = ["### SAE FEATURES TO CLUSTER ###"]
    
    feature_nodes = [n for n in graph_data.get("nodes", []) if n.get("type") == 'feature']
    
    for node in feature_nodes:
        # Assuming parse_act_html and parse_logits_html are available in your scope
        act_text = parse_act_html(node.get("act_html", ""), include_full_context)
        logit_text = parse_logits_html(node.get("logits_html", ""))
        
        output.append(f"\nFeature ID: {node['id']}")
        output.append(f"Layer: {node.get('layer', 'N/A')} | Seq Pos: {node.get('seq', 'N/A')}")
        output.append(f"Act Context:\n{act_text}")
        output.append(f"Logits (Promotes/Suppresses):\n  {logit_text}")

    return "\n".join(output)

def get_llm_clustering_prompt(n_clusters: int | None = None) -> str:
    """
    Generates the system prompt for the LLM. 
    Dynamically injects a strict rule if a specific number of clusters is requested.
    """
    
    if n_clusters is not None:
        cluster_rule = f"You MUST group the features into EXACTLY {n_clusters} clusters. Number the cluster_id from 0 to {n_clusters - 1}."
    else:
        cluster_rule = "You can determine the optimal number of clusters based on the semantic grouping of the data."

    return f"""You are an expert AI interpretability researcher. Your task is to analyze a list of Sparse Autoencoder (SAE) features extracted from a transformer language model and group them into semantically meaningful clusters.

DATA PROVIDED:
- Feature ID: Unique identifier for the feature.
- Act Context: Shows text intervals where the feature fired (activating tokens). THIS IS THE GOLD STANDARD for interpretation. Look for patterns (e.g., tracking plural nouns, specific concepts, or syntax).
- Logits: Shows tokens the feature promotes or suppresses. Use this to confirm the feature's meaning.

YOUR TASK:
1. Analyze the "Act Context" and "Logits" for every feature provided.
2. Write a short, general 1-sentence caption explaining what each feature detects or represents.
3. Group features that track similar concepts, syntactical roles, or semantic meanings into the same cluster. 
4. Output the result STRICTLY as a JSON object matching the schema below. Do not include any markdown formatting outside the JSON block.

CRITICAL RULES:
- EVERY feature ID provided in the prompt MUST be assigned to exactly one cluster. Do not leave any feature behind.
- {cluster_rule}
- Each cluster has AT LEAST one feature.
- Return ONLY valid JSON.

JSON SCHEMA:
{{
    "clusters": [
        {{
            "cluster_id": 0,
            "cluster_caption": "Features that track plural nouns and counting",
            "feature_ids": ["layer_2::seq_4::feat::123", "layer_5::seq_4::feat::456"]
        }}
    ]
}}
"""

from openai import OpenAI

def call_llm_for_clustering(api_key: str, system_prompt: str, user_content: str) -> str:
    """
    Wrapper to call the LLM using the specific custom endpoint configuration.
    Forces JSON output and uses 0.0 temperature for deterministic clustering.
    """
    # Configured to match your evaluation setup
    # LLMSEL_URL points at any OpenAI-compatible endpoint serving "gpt-oss-120b" (e.g. the vLLM
    # server scripts/circuit_compress_eval.sh starts with LLMSEL_SELFHOST=true); unset = the
    # University of Florida gateway the paper used.
    import os
    local_url = os.environ.get("LLMSEL_URL")
    client = OpenAI(
        api_key=api_key or "EMPTY", 
        base_url=local_url or "https://api.ai.it.ufl.edu/v1", 
        timeout=300.0,  # was 100: large circuits on a busy gateway timed out -> silent FeatFlow fallback
        max_retries=0
    )
    
    # Retry transient gateway failures (429 per-key parallel/rate caps, timeouts, 5xx, connection).
    # Without this every such failure silently became a FeatFlow clustering in run_clustering_llm.
    # A 400 (e.g. prompt over the 128k context) is not retried: it falls back as in the paper code.
    import time, random
    from openai import RateLimitError, APITimeoutError, APIConnectionError, InternalServerError
    for attempt in range(1, 16):
        try:
            response = client.chat.completions.create(
                model="gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0,
                response_format={ "type": "json_object" }
            )
            break
        except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) as e:
            if attempt == 15:
                raise
            wait = min(60.0, 2.0 ** attempt) * (0.5 + random.random())
            print(f"[call_llm_for_clustering] attempt {attempt} failed ({type(e).__name__}); retry in {wait:.0f}s")
            time.sleep(wait)
    
    return response.choices[0].message.content # type: ignore