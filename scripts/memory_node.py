#!/usr/bin/env python3

import rospy
import json
from similarity_memory.srv import ProcessData, ProcessDataResponse
from similarity_memory.srv import LLMQuery

# Import the class from our local src module
from similarity_memory.core import SimilarityMemoryCache

# --- Global Cache & LLM Client ---
g_cache = None
g_llm_service_proxy = None

def handle_process_data(req):
    global g_cache, g_llm_service_proxy

    try:
        # --- BRANCH 1: "Add/Forward" Request ---
        if req.value_json:
            rospy.loginfo("Received 'Add/Forward' request.")
            try:
                parsed_value = json.loads(req.value_json)
                data_to_store = parsed_value["data"]
                available_to_llm = parsed_value["available_to_llm"]
                data_json_string = json.dumps(data_to_store)
            except Exception as e:
                return ProcessDataResponse(success=False, cache_hit=False, response_data=f"Invalid JSON: {e}")

            if available_to_llm == False:
                rospy.loginfo(f"Caching key: '{req.key}'")
                g_cache.add(req.key, data_json_string)
                return ProcessDataResponse(success=True, cache_hit=False, response_data="Data cached.")
            else:
                rospy.loginfo("Forwarding directly to LLM...")
                llm_response = g_llm_service_proxy(query=req.key, context=data_json_string)
                return ProcessDataResponse(success=True, cache_hit=False, response_data=llm_response.answer)

        # --- BRANCH 2: "Query" Request ---
        else:
            rospy.loginfo(f"Received 'Query' request for: '{req.key}'")
            retrieved_json_string, score = g_cache.query(req.key)

            if retrieved_json_string is not None:
                rospy.loginfo(f"Cache HIT (Score: {score:.4f})")
                return ProcessDataResponse(success=True, cache_hit=True, response_data=retrieved_json_string)
            else:
                rospy.loginfo("Cache MISS. Forwarding to LLM.")
                llm_response = g_llm_service_proxy(query=req.key, context="")
                
                rospy.loginfo("Caching new LLM response.")
                g_cache.add(req.key, llm_response.answer)
                return ProcessDataResponse(success=True, cache_hit=False, response_data=llm_response.answer)

    except Exception as e:
        rospy.logerr(f"Error: {e}")
        return ProcessDataResponse(success=False, cache_hit=False, response_data=str(e))


def main_server():
    global g_cache, g_llm_service_proxy

    rospy.init_node('similarity_memory_node')
    
    # Parameters
    model_name = rospy.get_param('~model_name', 'all-MiniLM-L6-v2')
    threshold = rospy.get_param('~threshold', 0.6)
    max_size = rospy.get_param('~max_size', 100)
    llm_service_name = rospy.get_param('~llm_service_name', '/llm_service')

    # Init Cache
    g_cache = SimilarityMemoryCache(threshold=threshold, max_size=max_size, model_name=model_name)

    # Connect to LLM
    rospy.loginfo(f"Waiting for LLM service at '{llm_service_name}'...")
    rospy.wait_for_service(llm_service_name)
    g_llm_service_proxy = rospy.ServiceProxy(llm_service_name, LLMQuery)

    # Advertise Service
    s = rospy.Service('process_data', ProcessData, handle_process_data)
    rospy.loginfo("Similarity Memory Node is ready.")
    rospy.spin()

if __name__ == "__main__":
    main_server()