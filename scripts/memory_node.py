#!/usr/bin/env python3

import rospy
import json
import actionlib 
import os
import csv
from datetime import datetime

from similarity_memory.srv import ProcessData, ProcessDataResponse, MemoryQuery, MemoryQueryResponse
from elmira.msg import PromptTextLLMAction, PromptTextLLMGoal
from similarity_memory.core import SimilarityMemoryCache

g_cache = None
g_llm_client = None  
g_query_log_path = None

def call_llm_action(prompt_text):
    global g_llm_client

    goal = PromptTextLLMGoal()
    goal.prompt = prompt_text
    
    rospy.loginfo(f"Sending goal to LLM Action: {prompt_text[:50]}...")
    
    g_llm_client.send_goal(goal)
    
    # Wait for the action to finish (timeout is optional )
    finished = g_llm_client.wait_for_result(rospy.Duration(60.0))
    
    if not finished:
        return "Error: LLM Action timed out."
    
    result = g_llm_client.get_result()
    return result.response

def forward_hint_to_llm(key: str, hint_payload: str) -> str:
    """Format and forward hint/context information to the LLM."""
    prompt = (
        f"New hint for key '{key}':\n{hint_payload}\n\n"
        "Please acknowledge receipt of this hint."
    )
    return call_llm_action(prompt)

def handle_process_data(req):
    global g_cache, g_llm_client

    try:
        if req.hint:
            rospy.loginfo("Received hint payload.")
            payload = req.hint
            try:
                payload_obj = json.loads(req.hint)
                payload = json.dumps(payload_obj)
            except json.JSONDecodeError:
                pass

            rospy.loginfo(f"Caching hint under key '{req.key}'")
            g_cache.add(req.key, payload)

            if req.flag:
                rospy.loginfo("Flag true: forwarding freshly cached hint to LLM.")
                llm_response_text = forward_hint_to_llm(req.key, payload)
                return ProcessDataResponse(success=True, response=llm_response_text)
            else:
                return ProcessDataResponse(success=True, response="Hint cached.")

        elif req.key and req.flag:
            rospy.loginfo(f"Flag true without hint for key '{req.key}'. Attempting to forward cached hint.")
            cached_value, _ = g_cache.query(req.key)
            if cached_value is None:
                return ProcessDataResponse(success=False, response=f"No cached hint available for '{req.key}'.")

            llm_response_text = forward_hint_to_llm(req.key, cached_value)
            return ProcessDataResponse(success=True, response=llm_response_text)

        else:
            rospy.loginfo(f"Received Unknown request for key: '{req.key}' without hint or flag. returning if some value was cached.")
            retrieved_json_string, score = g_cache.query(req.key)

            if retrieved_json_string is not None:
                rospy.loginfo(f"Cache HIT (Score: {score:.4f})")
                return ProcessDataResponse(success=True, response=retrieved_json_string)
            else:
                rospy.loginfo("Cache MISS. Returning empty response.")
                return ProcessDataResponse(success=True, response="")

    except Exception as e:
        rospy.logerr(f"Error: {e}")
        return ProcessDataResponse(success=False, response=str(e))

def log_query_event(query_text, success, similarity_score, answer_text):
    global g_query_log_path

    if not g_query_log_path:
        return

    try:
        directory = os.path.dirname(g_query_log_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        file_exists = os.path.isfile(g_query_log_path)
        with open(g_query_log_path, 'a', newline='') as log_file:
            writer = csv.writer(log_file)
            if not file_exists:
                writer.writerow(["timestamp", "query", "success", "similarity", "answer"])
            writer.writerow([
                datetime.utcnow().isoformat(),
                query_text,
                bool(success),
                float(similarity_score),
                answer_text
            ])
    except Exception as log_err:
        rospy.logwarn(f"Failed to log query event: {log_err}")

def handle_llm_query(req):
    global g_cache

    answer_text = ""
    similarity_score = 0.0
    success = False

    try:
        if not req.key:
            raise ValueError("Request key cannot be empty.")

        cached_value, similarity_score = g_cache.query(req.key)
        if cached_value is not None:
            rospy.loginfo(f"LLM query cache HIT for '{req.key}' (Similarity: {similarity_score:.4f})")
            answer_text = cached_value
            success = True
        else:
            rospy.loginfo(
                f"LLM query cache MISS for '{req.key}'. Best similarity was {similarity_score:.4f} (threshold {g_cache.threshold:.2f})."
            )
            answer_text = ""
            success = False

        return MemoryQueryResponse(success=success, answer=answer_text, similarity_score=similarity_score)

    except Exception as e:
        rospy.logerr(f"LLM query error: {e}")
        success = False
        answer_text = str(e)
        similarity_score = 0.0
        return MemoryQueryResponse(success=False, answer=answer_text, similarity_score=similarity_score)

    finally:
        log_query_event(req.key, success, similarity_score, answer_text)

def main_server():
    global g_cache, g_llm_client, g_query_log_path

    rospy.init_node('similarity_memory_node')
    
    model_name = rospy.get_param('~model_name', 'all-MiniLM-L6-v2')
    threshold = rospy.get_param('~threshold', 0.6)
    max_size = rospy.get_param('~max_size', 100)
   
    llm_action_name = rospy.get_param('~llm_service_name', 'llm_chat') 
    default_log_filename = f"llm_query_log_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.csv"
    default_log_path = os.path.join(os.getcwd(), default_log_filename)
    g_query_log_path = rospy.get_param('~llm_query_log_path', default_log_path)
    rospy.loginfo(f"LLM query log file: {g_query_log_path}")

    g_cache = SimilarityMemoryCache(threshold=threshold, max_size=max_size, model_name=model_name)
    rospy.loginfo(f"Waiting for LLM Action Server at '{llm_action_name}'...")
    
    g_llm_client = actionlib.SimpleActionClient(llm_action_name, PromptTextLLMAction)
    
    g_llm_client.wait_for_server()

    rospy.loginfo(f"Connected to LLM Action Server at '{llm_action_name}'.")
    s = rospy.Service('memory_service', ProcessData, handle_process_data)
    q = rospy.Service('llm_query', MemoryQuery, handle_llm_query)
    rospy.loginfo("Similarity Memory Node is ready (Services: memory_service, llm_query).")
    rospy.spin()

if __name__ == "__main__":
    main_server()