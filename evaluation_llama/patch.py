def draft_only_forward(input_ids, model, tokenizer, max_new_tokens, statistics=None, optimizer=None, utility=None,
                  logits_processor=None, max_steps=512):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    input_ids = input_ids.clone()
    
    (
        past_key_values,
        past_key_values_data,
        current_length_data,
    ) = initialize_past_key_values(model.model)
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data
    
    model.draft_kv_compress = statistics.get("draft_kv_compress", False)
    model.draft_kv_retain_ratio = statistics.get("draft_kv_retain_ratio", 1.0)
    
    reset_swift_mode(model)
    
    with torch.inference_mode():
        outputs, logits = swift_verify(model, input_ids, past_key_values=past_key_values)
        if logits_processor is not None:
            last_logits = logits[:, -1]
            last_logits = logits_processor(None, last_logits)
            probabilities = torch.nn.functional.softmax(last_logits, dim=1)
            sample_token = torch.multinomial(probabilities, 1)
        else:
            sample_token = torch.argmax(logits[:, -1])[None, None]
            
        swift_logits, top1_prob = swift_draft(
            model,
            input_ids=sample_token,
            full_input_ids=input_ids,
            new_token_num=0,
            past_key_values_data=past_key_values_data,
            current_length_data=current_length_data,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
            max_step_draft=max_new_tokens,
            stop_threshold=-1.0,
        )
        
    (ss_token, ss_prob, ss_op) = swift_logits
    drafted_tokens = []
    for step_token in ss_token:
        if len(step_token.shape) == 3:
            tok = step_token[0, 0, 0].item()
        else:
            tok = step_token[0, 0].item()
        drafted_tokens.append(tok)
        if tok == tokenizer.eos_token_id:
            break
            
    drafted_tensor = torch.tensor([drafted_tokens], dtype=torch.long, device=input_ids.device)
    output_ids = torch.cat([input_ids, sample_token, drafted_tensor], dim=1)
    
    new_token_num = output_ids.shape[1] - input_ids.shape[1]
    import numpy as np
    accept_length_list = [1] * new_token_num
    
    return output_ids, new_token_num, new_token_num, accept_length_list, new_token_num
