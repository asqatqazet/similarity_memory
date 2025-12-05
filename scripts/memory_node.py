#!/usr/bin/env python3

import rospy
import json
import actionlib 

from similarity_memory.srv import ProcessData, ProcessDataResponse
from elmira.msg import PromptTextLLMAction, PromptTextLLMGoal
from similarity_memory.core import SimilarityMemoryCache

g_cache = None
g_llm_client = None  

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

def handle_process_data(req):
    global g_cache, g_llm_client

    try:
        if req.hint:
            rospy.loginfo("Received 'Add/Forward' request.")
            payload = req.hint
            # If the hint looks like JSON, keep a canonical string version; otherwise store raw text
            try:
                payload_obj = json.loads(req.hint)
                payload = json.dumps(payload_obj)
            except json.JSONDecodeError:
                pass

            if not req.flag:
                rospy.loginfo(f"Caching key: '{req.key}'")
                g_cache.add(req.key, payload)
                return ProcessDataResponse(success=True, response="Data cached.")
            else:
                rospy.loginfo("Forwarding directly to LLM...")
                full_prompt = f"Context Data: {payload}\n\nUser Question: {req.key}"
                llm_response_text = call_llm_action(full_prompt)
                return ProcessDataResponse(success=True, response=llm_response_text)

        else:
            rospy.loginfo(f"Received 'Query' request for: '{req.key}'")
            retrieved_json_string, score = g_cache.query(req.key)

            if retrieved_json_string is not None:
                rospy.loginfo(f"Cache HIT (Score: {score:.4f})")
                return ProcessDataResponse(success=True, response=retrieved_json_string)
            else:
                rospy.loginfo("Cache MISS. Forwarding to LLM.")
                llm_response_text = call_llm_action(req.key)
                rospy.loginfo("Caching new LLM response.")
                g_cache.add(req.key, llm_response_text)
                return ProcessDataResponse(success=True, response=llm_response_text)

    except Exception as e:
        rospy.logerr(f"Error: {e}")
        return ProcessDataResponse(success=False, response=str(e))

def main_server():
    global g_cache, g_llm_client

    rospy.init_node('similarity_memory_node')
    
    model_name = rospy.get_param('~model_name', 'all-MiniLM-L6-v2')
    threshold = rospy.get_param('~threshold', 0.6)
    max_size = rospy.get_param('~max_size', 100)
   
    llm_action_name = rospy.get_param('~llm_service_name', 'llm_chat') 

    g_cache = SimilarityMemoryCache(threshold=threshold, max_size=max_size, model_name=model_name)
    rospy.loginfo(f"Waiting for LLM Action Server at '{llm_action_name}'...")
    
    g_llm_client = actionlib.SimpleActionClient(llm_action_name, PromptTextLLMAction)
    
    g_llm_client.wait_for_server()

    rospy.loginfo(f"Connected to LLM Action Server at '{llm_action_name}'.")
    s = rospy.Service('memory_service', ProcessData, handle_process_data)
    rospy.loginfo("Similarity Memory Node is ready (Service: memory_service).")
    rospy.spin()

if __name__ == "__main__":
    main_server()