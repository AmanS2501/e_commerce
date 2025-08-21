import json
import os
import requests
from core.groq_llm import GroqLLM
from inputimeout import inputimeout, TimeoutOccurred

DATA_FOLDER = "data"
ALLMART_API_URL = "https://api.allmart.fashion/api/v1/products"
TIMEOUT_SECONDS = 120

SYSTEM_PROMPT = """
You are a helpful e-commerce assistant for the Allmart platform. For every user message, always reply ONLY in valid JSON specifying the action and parameters.
Possible actions: show_cart, add_to_cart, remove_from_cart, search_products, get_order_status, reply.
For add_to_cart/remove_from_cart, always return product_id as "id".
For search_products, always return filters as a dictionary under the "filters" key, e.g. "filters":{"category": "electronics", "max_price": 40000}
For get_order_status, return order_id.
If user just wants to chat, return: {"action": "reply", "message": "..."}.
DO NOT EXPLAIN YOUR THINKING. Always reply with only the JSON.
"""

def load_cart(user_id):
    path = os.path.join(DATA_FOLDER, f"cart_{user_id}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

def save_cart(user_id, cart):
    path = os.path.join(DATA_FOLDER, f"cart_{user_id}.json")
    with open(path, "w") as f:
        json.dump(cart, f, indent=2)

def fetch_products(filters, page=1, limit=12):
    params = {
        "page": page,
        "limit": limit,
        "sortBy": "averageRating",
        "sortOrder": "desc",
        "inStock": "true"
    }
    # Only send categorySlug if category is set and is not "all"
    category = filters.get("category")
    if category and category.lower() != "all":
        params["categorySlug"] = category
    if "min_price" in filters:
        params["minPrice"] = filters["min_price"]
    if "max_price" in filters:
        params["maxPrice"] = filters["max_price"]
    try:
        resp = requests.get(ALLMART_API_URL, params=params, timeout=10)
        data = resp.json()
    except Exception as e:
        print("Product API error:", e)
        return []
    products = (
        data.get("data", {}).get("data") or 
        data.get("data", []) or 
        data.get("products", []) or []
    )
    return products


def print_products(products):
    if not products:
        print("(No products found.)")
        return
    for p in products:
        prod_id = p.get('id', 'N/A')
        name = p.get("title", "Unknown")
        price = p.get("price", "N/A")
        brand = p.get("brand", "")
        print(f"{prod_id}: {name} (₹{price})", f"[{brand}]" if brand else "")

def extract_filters(llm_response):
    filters = llm_response.get("filters", {}).copy()
    # Accept filter keys at top level as well
    if "category" in llm_response:
        filters["category"] = llm_response["category"]
    if "max_price" in llm_response:
        filters["max_price"] = llm_response["max_price"]
    if "min_price" in llm_response:
        filters["min_price"] = llm_response["min_price"]
    if "price_range" in llm_response:
        pr = llm_response["price_range"]
        if "max" in pr:
            filters["max_price"] = pr["max"]
        if "min" in pr:
            filters["min_price"] = pr["min"]
    return filters

def main():
    os.makedirs(DATA_FOLDER, exist_ok=True)
    user_id = input("Enter your user_id: ").strip()
    llm = GroqLLM()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    print(
        "Welcome! Example: 'show my cart', 'add <product_id> to cart', "
        "'show electronics under 40000'.\n"
        "Type 'exit' to quit."
    )
    while True:
        try:
            user_input = inputimeout(prompt="You: ", timeout=TIMEOUT_SECONDS)
        except TimeoutOccurred:
            print("\nSession timed out due to inactivity. Exiting.")
            break
        if user_input.strip().lower() == "exit":
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})
        llm_response = llm.generate(messages)
        print(f"LLM (JSON): {llm_response}")
        # try:
        #     res = json.loads(llm_response)
        # except Exception:
        #     print("LLM did not return valid JSON. Please try again.")
        #     continue

        # action = res.get("action")
        # if action == "show_cart":
        #     cart = load_cart(user_id)
        #     if not cart:
        #         print("Your cart is empty.")
        #     else:
        #         print("Your cart:")
        #         print_products(cart)

        # elif action == "add_to_cart":
        #     product_id = res.get("product_id") or res.get("id")
        #     if not product_id:
        #         print("No product_id specified by LLM.")
        #         continue
        #     # fetch current popular products (first page), then match the ID
        #     products = fetch_products({}, page=1, limit=50)
        #     product = next((p for p in products if p.get('id') == product_id), None)
        #     if not product:
        #         print("Invalid product_id (not found in first 50 results).")
        #     else:
        #         cart = load_cart(user_id)
        #         if any(item.get('id') == product_id for item in cart):
        #             print("Already in cart.")
        #         else:
        #             cart.append(product)
        #             save_cart(user_id, cart)
        #             print(f"Added to cart: {product.get('title','(No Name)')}")

        # elif action == "remove_from_cart":
        #     product_id = res.get("product_id") or res.get("id")
        #     cart = load_cart(user_id)
        #     new_cart = [item for item in cart if item.get('id') != product_id]
        #     save_cart(user_id, new_cart)
        #     if len(cart) == len(new_cart):
        #         print("Product not in cart.")
        #     else:
        #         print(f"Removed product {product_id} from cart.")

        # elif action == "search_products":
        #     filters = extract_filters(res)
        #     # Default to 12 products per page
        #     products = fetch_products(filters, page=1, limit=12)
        #     print("Products matching filters:")
        #     print_products(products)

        # elif action == "reply":
        #     print(f"Bot: {res.get('message','')}")
        # else:
        #     print("Unknown action. LLM output:", res)

        # messages.append({"role": "assistant", "content": llm_response})

if __name__ == "__main__":
    main()
