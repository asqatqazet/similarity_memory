#!/usr/bin/env python3

import rospy
import json
import actionlib
from similarity_memory.srv import ProcessData, ProcessDataResponse

# --- Import ELMiRA Action Definitions ---
from similarity_memory.msg import PromptTextLLMAction, PromptTextLLMGoal

# Import the class from our local src module
from similarity_memory.core import SimilarityMemoryCache

# --- Global Cache & LLM Client ---
g_cache = None
g_llm_client = None

# --- Pre-script Templates ---
TEMPLATE_WITH_CONTEXT = (
    "I am providing you with hint for the current game status.\n"
    "Hint is: {context}\n"
    "Key word was: {query}"
)

def extract_speech_from_json(json_str):
    """
    Parses the GPT4Server JSON response to find the 'speak' action.
    """
    try:
        data = json.loads(json_str)
        actions = data.get('actions', [])
        for action in actions:
            if action.get('mode') == 'speak':
                return action.get('text', "")
        return json_str 
    except json.JSONDecodeError:
        return json_str

def call_llm_action(prompt_text):
    """Helper to send goal to Action Server"""
    global g_llm_client
    
    goal = PromptTextLLMGoal()
    goal.prompt = prompt_text
    
    g_llm_client.send_goal(goal)
    finished = g_llm_client.wait_for_result(rospy.Duration(60.0))
    
    if not finished:
        return "Error: LLM timed out."
    
    result = g_llm_client.get_result()
    return extract_speech_from_json(result.response)

def handle_process_data(req):
    global g_cache

    try:
        # -----------------------------------------------------------------
        # BRANCH 1: "Add/Forward" Request (Hint is NOT empty)
        # -----------------------------------------------------------------
        if req.hint and req.hint.strip():
            rospy.loginfo("Received 'Add/Forward' request (Hint present).")
            
            # --- Logic A: Flag is FALSE (Store only) ---
            if req.flag == False:
                rospy.loginfo(f"Flag is FALSE. Caching key: '{req.key}' with hint.")
                g_cache.add(req.key, req.hint)
                return ProcessDataResponse(success=True, response_data="Data cached.")
            
            # --- Logic B: Flag is TRUE (Forward to LLM) ---
            else:
                rospy.loginfo("Flag is TRUE. Forwarding context to LLM...")
                final_prompt = TEMPLATE_WITH_CONTEXT.format(
                    context=req.hint, 
                    query=req.key
                )
                llm_response_text = call_llm_action(final_prompt)
                return ProcessDataResponse(success=True, response_data=llm_response_text)

        # -----------------------------------------------------------------
        # BRANCH 2: "Query" Request (Hint IS empty)
        # -----------------------------------------------------------------
        else:
            rospy.loginfo(f"Received 'Query' request for: '{req.key}'")
            retrieved_data, score = g_cache.query(req.key)

            if retrieved_data is not None:
                # --- CACHE HIT ---
                rospy.loginfo(f"Cache HIT (Score: {score:.4f})")
                return ProcessDataResponse(success=True, response_data=retrieved_data)
            else:
                # --- CACHE MISS ---
                # CHANGE: We do NOT call the LLM anymore. We just report failure.
                rospy.loginfo("Cache MISS. No matching memory found.")
                
                return ProcessDataResponse(
                    success=False,  # Signal to the client that we found nothing
                    response_data="" 
                )

    except Exception as e:
        rospy.logerr(f"Error in handle_process_data: {e}")
        return ProcessDataResponse(success=False, response_data=str(e))


def main_server():
    global g_cache, g_llm_client

    rospy.init_node('similarity_memory_node')
    
    # Parameters
    model_name = rospy.get_param('~model_name', 'all-MiniLM-L6-v2')
    threshold = rospy.get_param('~threshold', 0.6)
    max_size = rospy.get_param('~max_size', 100)
    llm_action_name = rospy.get_param('~llm_action_name', 'llm_chat') 

    # Init Cache
    g_cache = SimilarityMemoryCache(threshold=threshold, max_size=max_size, model_name=model_name)

    # Connect to LLM Action Server
    # (We still need this connection for Branch 1 Logic B)
    rospy.loginfo(f"Waiting for LLM Action Server at '{llm_action_name}'...")
    g_llm_client = actionlib.SimpleActionClient(llm_action_name, PromptTextLLMAction)
    
    if not g_llm_client.wait_for_server(rospy.Duration(20.0)):
        rospy.logerr(f"Could not connect to Action Server '{llm_action_name}'.")
        return

    # Advertise Service
    s = rospy.Service('memory_service', ProcessData, handle_process_data)
    rospy.loginfo("Similarity Memory Node is ready.")
    rospy.spin()

if __name__ == "__main__":
    main_server()