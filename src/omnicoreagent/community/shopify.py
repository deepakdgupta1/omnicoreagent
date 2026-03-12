"""
Shopify Tools for OmnicoreAgent

A toolkit for analyzing sales data, product performance, and customer insights
using the Shopify Admin GraphQL API.

Required scopes: read_orders, read_products, read_customers, read_analytics
"""

import json
from collections import Counter
from datetime import datetime, timedelta
from itertools import combinations
from os import getenv
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:
    httpx = None

from omnicoreagent.core.tools.local_tools_registry import Tool


class ShopifyTools:
    def __init__(
        self,
        shop_name: Optional[str] = None,
        access_token: Optional[str] = None,
        api_version: str = "2025-10",
        timeout: int = 30,
    ):
        if httpx is None:
            raise ImportError(
                "`httpx` not installed. Please install using `pip install httpx`"
            )
        self.shop_name = shop_name or getenv("SHOPIFY_SHOP_NAME")
        self.access_token = access_token or getenv("SHOPIFY_ACCESS_TOKEN")
        self.api_version = api_version
        self.timeout = timeout
        self.base_url = f"https://{self.shop_name}.myshopify.com/admin/api/{self.api_version}/graphql.json"

    def _make_graphql_request(
        self, query: str, variables: Optional[Dict] = None
    ) -> Dict:
        headers = {
            "X-Shopify-Access-Token": self.access_token or "",
            "Content-Type": "application/json",
        }
        body: Dict[str, Any] = {"query": query}
        if variables:
            body["variables"] = variables
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.base_url, headers=headers, json=body)
            try:
                result = response.json()
                if "errors" in result:
                    return {"error": result["errors"]}
                return result.get("data", {})
            except json.JSONDecodeError:
                return {"error": f"Failed to parse response: {response.text}"}

    def get_tool(self) -> Tool:
        return Tool(
            name="shopify_get_shop_info",
            description="Get basic information about the Shopify store.",
            inputSchema={"type": "object", "properties": {}},
            function=self._get_shop_info,
        )

    async def _get_shop_info(self) -> Dict[str, Any]:
        query = """
        query {
            shop {
                name
                email
                currencyCode
                primaryDomain { url }
                billingAddress { country city }
                plan { displayName }
            }
        }
        """
        result = self._make_graphql_request(query)
        if "error" in result:
            return {"status": "error", "data": None, "message": str(result["error"])}
        return {
            "status": "success",
            "data": result.get("shop", {}),
            "message": "Shop info retrieved",
        }


class ShopifyGetProducts(ShopifyTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="shopify_get_products",
            description="Get products from the Shopify store.",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_results": {"type": "integer", "default": 50},
                    "status": {
                        "type": "string",
                        "description": "ACTIVE, ARCHIVED, or DRAFT",
                    },
                },
            },
            function=self._get_products,
        )

    async def _get_products(
        self, max_results: int = 50, status: Optional[str] = None
    ) -> Dict[str, Any]:
        query_filter = f', query: "status:{status}"' if status else ""
        query = f"""
        query {{
            products(first: {min(max_results, 250)}{query_filter}) {{
                edges {{
                    node {{
                        id title status totalInventory createdAt updatedAt
                        priceRangeV2 {{
                            minVariantPrice {{ amount currencyCode }}
                            maxVariantPrice {{ amount currencyCode }}
                        }}
                        variants(first: 10) {{
                            edges {{ node {{ id title sku price inventoryQuantity }} }}
                        }}
                    }}
                }}
            }}
        }}
        """
        result = self._make_graphql_request(query)
        if "error" in result:
            return {"status": "error", "data": None, "message": str(result["error"])}
        products = []
        for edge in result.get("products", {}).get("edges", []):
            node = edge["node"]
            products.append(
                {
                    "id": node["id"],
                    "title": node["title"],
                    "status": node["status"],
                    "total_inventory": node.get("totalInventory"),
                    "created_at": node.get("createdAt"),
                    "price_range": {
                        "min": node.get("priceRangeV2", {})
                        .get("minVariantPrice", {})
                        .get("amount"),
                        "max": node.get("priceRangeV2", {})
                        .get("maxVariantPrice", {})
                        .get("amount"),
                        "currency": node.get("priceRangeV2", {})
                        .get("minVariantPrice", {})
                        .get("currencyCode"),
                    },
                    "variants": [
                        {
                            "id": v["node"]["id"],
                            "title": v["node"]["title"],
                            "sku": v["node"].get("sku"),
                            "price": v["node"]["price"],
                            "inventory": v["node"].get("inventoryQuantity"),
                        }
                        for v in node.get("variants", {}).get("edges", [])
                    ],
                }
            )
        return {
            "status": "success",
            "data": products,
            "message": f"Found {len(products)} products",
        }


class ShopifyGetOrders(ShopifyTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="shopify_get_orders",
            description="Get recent orders from the Shopify store.",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_results": {"type": "integer", "default": 50},
                    "status": {
                        "type": "string",
                        "description": "paid, pending, refunded",
                    },
                    "created_after": {"type": "string", "description": "YYYY-MM-DD"},
                    "created_before": {"type": "string", "description": "YYYY-MM-DD"},
                },
            },
            function=self._get_orders,
        )

    async def _get_orders(
        self,
        max_results: int = 50,
        status: Optional[str] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
    ) -> Dict[str, Any]:
        query_parts = []
        if created_after:
            query_parts.append(f"created_at:>={created_after}")
        if created_before:
            query_parts.append(f"created_at:<={created_before}")
        if status:
            query_parts.append(f"financial_status:{status}")
        query_filter = " AND ".join(query_parts) if query_parts else ""
        query_param = f', query: "{query_filter}"' if query_filter else ""

        query = f"""
        query {{
            orders(first: {min(max_results, 250)}{query_param}, sortKey: CREATED_AT, reverse: true) {{
                edges {{
                    node {{
                        id name createdAt displayFinancialStatus displayFulfillmentStatus
                        totalPriceSet {{ shopMoney {{ amount currencyCode }} }}
                        subtotalPriceSet {{ shopMoney {{ amount }} }}
                        customer {{ id email firstName lastName }}
                        lineItems(first: 50) {{
                            edges {{
                                node {{
                                    id title quantity
                                    variant {{ id sku }}
                                    originalUnitPriceSet {{ shopMoney {{ amount }} }}
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
        """
        result = self._make_graphql_request(query)
        if "error" in result:
            return {"status": "error", "data": None, "message": str(result["error"])}
        orders = []
        for edge in result.get("orders", {}).get("edges", []):
            node = edge["node"]
            customer = node.get("customer") or {}
            orders.append(
                {
                    "id": node["id"],
                    "name": node["name"],
                    "created_at": node["createdAt"],
                    "financial_status": node.get("displayFinancialStatus"),
                    "fulfillment_status": node.get("displayFulfillmentStatus"),
                    "total": node.get("totalPriceSet", {})
                    .get("shopMoney", {})
                    .get("amount"),
                    "subtotal": node.get("subtotalPriceSet", {})
                    .get("shopMoney", {})
                    .get("amount"),
                    "currency": node.get("totalPriceSet", {})
                    .get("shopMoney", {})
                    .get("currencyCode"),
                    "customer": {
                        "id": customer.get("id"),
                        "email": customer.get("email"),
                        "name": f"{customer.get('firstName', '')} {customer.get('lastName', '')}".strip(),
                    }
                    if customer
                    else None,
                    "line_items": [
                        {
                            "id": item["node"]["id"],
                            "title": item["node"]["title"],
                            "quantity": item["node"]["quantity"],
                            "unit_price": item["node"]
                            .get("originalUnitPriceSet", {})
                            .get("shopMoney", {})
                            .get("amount"),
                            "sku": item["node"].get("variant", {}).get("sku")
                            if item["node"].get("variant")
                            else None,
                        }
                        for item in node.get("lineItems", {}).get("edges", [])
                    ],
                }
            )
        return {
            "status": "success",
            "data": orders,
            "message": f"Found {len(orders)} orders",
        }


class ShopifyGetTopSellingProducts(ShopifyTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="shopify_get_top_selling_products",
            description="Get the top selling products by quantity sold.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 10},
                    "created_after": {"type": "string"},
                    "created_before": {"type": "string"},
                },
            },
            function=self._get_top_selling,
        )

    async def _get_top_selling(
        self,
        limit: int = 10,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
    ) -> Dict[str, Any]:
        query_parts = ["financial_status:paid"]
        if created_after:
            query_parts.append(f"created_at:>={created_after}")
        if created_before:
            query_parts.append(f"created_at:<={created_before}")
        query_filter = " AND ".join(query_parts)

        query = f"""
        query {{
            orders(first: 250, query: "{query_filter}", sortKey: CREATED_AT) {{
                edges {{
                    node {{
                        lineItems(first: 100) {{
                            edges {{
                                node {{
                                    title quantity
                                    variant {{ id product {{ id title }} }}
                                    originalUnitPriceSet {{ shopMoney {{ amount }} }}
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
        """
        result = self._make_graphql_request(query)
        if "error" in result:
            return {"status": "error", "data": None, "message": str(result["error"])}

        product_sales: Dict[str, Dict[str, Any]] = {}
        for order_edge in result.get("orders", {}).get("edges", []):
            for item_edge in order_edge["node"].get("lineItems", {}).get("edges", []):
                item = item_edge["node"]
                variant = item.get("variant")
                if not variant or not variant.get("product"):
                    continue
                pid = variant["product"]["id"]
                qty = item["quantity"]
                price = float(
                    item.get("originalUnitPriceSet", {})
                    .get("shopMoney", {})
                    .get("amount", 0)
                )
                if pid not in product_sales:
                    product_sales[pid] = {
                        "id": pid,
                        "title": variant["product"]["title"],
                        "total_quantity": 0,
                        "total_revenue": 0.0,
                        "order_count": 0,
                    }
                product_sales[pid]["total_quantity"] += qty
                product_sales[pid]["total_revenue"] += qty * price
                product_sales[pid]["order_count"] += 1

        sorted_products = sorted(
            product_sales.values(), key=lambda x: x["total_quantity"], reverse=True
        )[:limit]
        for i, p in enumerate(sorted_products):
            p["rank"] = i + 1
            p["total_revenue"] = round(p["total_revenue"], 2)
        return {
            "status": "success",
            "data": sorted_products,
            "message": f"Top {len(sorted_products)} products",
        }


class ShopifyGetProductsBoughtTogether(ShopifyTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="shopify_get_products_bought_together",
            description="Find products frequently bought together.",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_occurrences": {"type": "integer", "default": 2},
                    "limit": {"type": "integer", "default": 20},
                    "created_after": {"type": "string"},
                    "created_before": {"type": "string"},
                },
            },
            function=self._get_bought_together,
        )

    async def _get_bought_together(
        self,
        min_occurrences: int = 2,
        limit: int = 20,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
    ) -> Dict[str, Any]:
        query_parts = ["financial_status:paid"]
        if created_after:
            query_parts.append(f"created_at:>={created_after}")
        if created_before:
            query_parts.append(f"created_at:<={created_before}")
        query_filter = " AND ".join(query_parts)

        query = f"""
        query {{
            orders(first: 250, query: "{query_filter}") {{
                edges {{
                    node {{
                        lineItems(first: 100) {{
                            edges {{ node {{ variant {{ product {{ id title }} }} }} }}
                        }}
                    }}
                }}
            }}
        }}
        """
        result = self._make_graphql_request(query)
        if "error" in result:
            return {"status": "error", "data": None, "message": str(result["error"])}

        pair_counter: Counter = Counter()
        product_info: Dict[str, str] = {}
        for order_edge in result.get("orders", {}).get("edges", []):
            products_in_order = []
            for item_edge in order_edge["node"].get("lineItems", {}).get("edges", []):
                variant = item_edge["node"].get("variant")
                if variant and variant.get("product"):
                    pid = variant["product"]["id"]
                    products_in_order.append(pid)
                    product_info[pid] = variant["product"]["title"]
            unique = list(set(products_in_order))
            if len(unique) >= 2:
                for pair in combinations(sorted(unique), 2):
                    pair_counter[pair] += 1

        pairs = [
            {
                "product_1": {"id": p[0], "title": product_info.get(p[0], "Unknown")},
                "product_2": {"id": p[1], "title": product_info.get(p[1], "Unknown")},
                "times_bought_together": c,
            }
            for p, c in pair_counter.most_common(limit)
            if c >= min_occurrences
        ]
        return {
            "status": "success",
            "data": pairs,
            "message": f"Found {len(pairs)} product pairs",
        }


class ShopifyGetSalesByDateRange(ShopifyTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="shopify_get_sales_by_date_range",
            description="Get sales summary for a specific date range.",
            inputSchema={
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["start_date", "end_date"],
            },
            function=self._get_sales_by_date,
        )

    async def _get_sales_by_date(
        self, start_date: str, end_date: str
    ) -> Dict[str, Any]:
        query = f"""
        query {{
            orders(first: 250, query: "created_at:>={start_date} AND created_at:<={end_date} AND financial_status:paid", sortKey: CREATED_AT) {{
                edges {{
                    node {{
                        createdAt
                        totalPriceSet {{ shopMoney {{ amount currencyCode }} }}
                        lineItems(first: 100) {{ edges {{ node {{ quantity }} }} }}
                    }}
                }}
            }}
        }}
        """
        result = self._make_graphql_request(query)
        if "error" in result:
            return {"status": "error", "data": None, "message": str(result["error"])}

        total_revenue = 0.0
        total_orders = 0
        total_items = 0
        daily: Dict[str, Dict[str, Any]] = {}
        currency = None

        for order_edge in result.get("orders", {}).get("edges", []):
            order = order_edge["node"]
            amount = float(
                order.get("totalPriceSet", {}).get("shopMoney", {}).get("amount", 0)
            )
            currency = (
                order.get("totalPriceSet", {}).get("shopMoney", {}).get("currencyCode")
            )
            date = order["createdAt"][:10]
            items = sum(
                i["node"]["quantity"]
                for i in order.get("lineItems", {}).get("edges", [])
            )
            total_revenue += amount
            total_orders += 1
            total_items += items
            if date not in daily:
                daily[date] = {"date": date, "revenue": 0.0, "orders": 0, "items": 0}
            daily[date]["revenue"] += amount
            daily[date]["orders"] += 1
            daily[date]["items"] += items

        sorted_daily = sorted(daily.values(), key=lambda x: x["date"])
        for d in sorted_daily:
            d["revenue"] = round(d["revenue"], 2)

        summary = {
            "period": {"start": start_date, "end": end_date},
            "total_revenue": round(total_revenue, 2),
            "total_orders": total_orders,
            "total_items_sold": total_items,
            "average_order_value": round(total_revenue / total_orders, 2)
            if total_orders > 0
            else 0,
            "currency": currency,
            "daily_breakdown": sorted_daily,
        }
        return {
            "status": "success",
            "data": summary,
            "message": f"Sales data for {start_date} to {end_date}",
        }


class ShopifyGetOrderAnalytics(ShopifyTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="shopify_get_order_analytics",
            description="Get comprehensive order analytics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "created_after": {"type": "string"},
                    "created_before": {"type": "string"},
                },
            },
            function=self._get_analytics,
        )

    async def _get_analytics(
        self, created_after: Optional[str] = None, created_before: Optional[str] = None
    ) -> Dict[str, Any]:
        query_parts = []
        if created_after:
            query_parts.append(f"created_at:>={created_after}")
        if created_before:
            query_parts.append(f"created_at:<={created_before}")
        query_filter = " AND ".join(query_parts) if query_parts else ""
        query_param = f', query: "{query_filter}"' if query_filter else ""

        query = f"""
        query {{
            orders(first: 250{query_param}, sortKey: CREATED_AT) {{
                edges {{
                    node {{
                        displayFinancialStatus displayFulfillmentStatus
                        totalPriceSet {{ shopMoney {{ amount currencyCode }} }}
                        subtotalPriceSet {{ shopMoney {{ amount }} }}
                        totalShippingPriceSet {{ shopMoney {{ amount }} }}
                        totalTaxSet {{ shopMoney {{ amount }} }}
                        lineItems(first: 100) {{ edges {{ node {{ quantity }} }} }}
                    }}
                }}
            }}
        }}
        """
        result = self._make_graphql_request(query)
        if "error" in result:
            return {"status": "error", "data": None, "message": str(result["error"])}

        orders = result.get("orders", {}).get("edges", [])
        if not orders:
            return {"status": "success", "data": None, "message": "No orders found"}

        total_orders = len(orders)
        total_revenue = total_subtotal = total_shipping = total_tax = 0.0
        total_items = 0
        currency = None
        financial_counts: Counter = Counter()
        fulfillment_counts: Counter = Counter()
        order_values: List[float] = []

        for oe in orders:
            o = oe["node"]
            amt = float(
                o.get("totalPriceSet", {}).get("shopMoney", {}).get("amount", 0)
            )
            currency = (
                o.get("totalPriceSet", {}).get("shopMoney", {}).get("currencyCode")
            )
            total_revenue += amt
            total_subtotal += float(
                o.get("subtotalPriceSet", {}).get("shopMoney", {}).get("amount", 0)
            )
            total_shipping += float(
                o.get("totalShippingPriceSet", {}).get("shopMoney", {}).get("amount", 0)
            )
            total_tax += float(
                o.get("totalTaxSet", {}).get("shopMoney", {}).get("amount", 0)
            )
            order_values.append(amt)
            items = sum(
                i["node"]["quantity"] for i in o.get("lineItems", {}).get("edges", [])
            )
            total_items += items
            financial_counts[o.get("displayFinancialStatus", "UNKNOWN")] += 1
            fulfillment_counts[o.get("displayFulfillmentStatus", "UNKNOWN")] += 1

        analytics = {
            "period": {
                "created_after": created_after,
                "created_before": created_before,
            },
            "total_orders": total_orders,
            "total_revenue": round(total_revenue, 2),
            "total_subtotal": round(total_subtotal, 2),
            "total_shipping": round(total_shipping, 2),
            "total_tax": round(total_tax, 2),
            "currency": currency,
            "average_order_value": round(total_revenue / total_orders, 2)
            if total_orders > 0
            else 0,
            "total_items_sold": total_items,
            "average_items_per_order": round(total_items / total_orders, 2)
            if total_orders > 0
            else 0,
            "min_order_value": round(min(order_values), 2) if order_values else 0,
            "max_order_value": round(max(order_values), 2) if order_values else 0,
            "financial_status_breakdown": dict(financial_counts),
            "fulfillment_status_breakdown": dict(fulfillment_counts),
        }
        return {
            "status": "success",
            "data": analytics,
            "message": "Order analytics retrieved",
        }


class ShopifyGetProductSalesBreakdown(ShopifyTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="shopify_get_product_sales_breakdown",
            description="Get detailed sales breakdown for a specific product.",
            inputSchema={
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "created_after": {"type": "string"},
                    "created_before": {"type": "string"},
                },
                "required": ["product_id"],
            },
            function=self._get_product_breakdown,
        )

    async def _get_product_breakdown(
        self,
        product_id: str,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not product_id.startswith("gid://"):
            product_id = f"gid://shopify/Product/{product_id}"

        query_parts = ["financial_status:paid"]
        if created_after:
            query_parts.append(f"created_at:>={created_after}")
        if created_before:
            query_parts.append(f"created_at:<={created_before}")
        query_filter = " AND ".join(query_parts)

        query = f"""
        query {{
            orders(first: 250, query: "{query_filter}") {{
                edges {{
                    node {{
                        createdAt
                        lineItems(first: 100) {{
                            edges {{
                                node {{
                                    title quantity
                                    variant {{ id title sku product {{ id title }} }}
                                    originalUnitPriceSet {{ shopMoney {{ amount currencyCode }} }}
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
        """
        result = self._make_graphql_request(query)
        if "error" in result:
            return {"status": "error", "data": None, "message": str(result["error"])}

        product_title = None
        total_qty = 0
        total_rev = 0.0
        order_count = 0
        currency = None
        variant_breakdown: Dict[str, Dict[str, Any]] = {}
        daily_sales: Dict[str, Dict[str, Any]] = {}

        for oe in result.get("orders", {}).get("edges", []):
            order = oe["node"]
            order_date = order["createdAt"][:10]
            found = False
            for ie in order.get("lineItems", {}).get("edges", []):
                item = ie["node"]
                variant = item.get("variant")
                if not variant or not variant.get("product"):
                    continue
                if variant["product"]["id"] == product_id:
                    product_title = variant["product"]["title"]
                    qty = item["quantity"]
                    price = float(
                        item.get("originalUnitPriceSet", {})
                        .get("shopMoney", {})
                        .get("amount", 0)
                    )
                    currency = (
                        item.get("originalUnitPriceSet", {})
                        .get("shopMoney", {})
                        .get("currencyCode")
                    )
                    vid = variant["id"]
                    total_qty += qty
                    total_rev += qty * price
                    found = True
                    if vid not in variant_breakdown:
                        variant_breakdown[vid] = {
                            "variant_id": vid,
                            "variant_title": variant.get("title", "Default"),
                            "sku": variant.get("sku"),
                            "quantity": 0,
                            "revenue": 0.0,
                        }
                    variant_breakdown[vid]["quantity"] += qty
                    variant_breakdown[vid]["revenue"] += qty * price
                    if order_date not in daily_sales:
                        daily_sales[order_date] = {
                            "date": order_date,
                            "quantity": 0,
                            "revenue": 0.0,
                        }
                    daily_sales[order_date]["quantity"] += qty
                    daily_sales[order_date]["revenue"] += qty * price
            if found:
                order_count += 1

        if product_title is None:
            return {
                "status": "error",
                "data": None,
                "message": "Product not found in any orders",
            }

        for v in variant_breakdown.values():
            v["revenue"] = round(v["revenue"], 2)
        sorted_daily = sorted(daily_sales.values(), key=lambda x: x["date"])
        for d in sorted_daily:
            d["revenue"] = round(d["revenue"], 2)

        data = {
            "product_id": product_id,
            "product_title": product_title,
            "total_quantity_sold": total_qty,
            "total_revenue": round(total_rev, 2),
            "order_count": order_count,
            "currency": currency,
            "average_units_per_order": round(total_qty / order_count, 2)
            if order_count > 0
            else 0,
            "variant_breakdown": list(variant_breakdown.values()),
            "daily_sales": sorted_daily,
        }
        return {
            "status": "success",
            "data": data,
            "message": f"Sales breakdown for {product_title}",
        }


class ShopifyGetCustomerOrderHistory(ShopifyTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="shopify_get_customer_order_history",
            description="Get order history for a specific customer.",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_email": {"type": "string"},
                    "max_orders": {"type": "integer", "default": 50},
                },
                "required": ["customer_email"],
            },
            function=self._get_customer_history,
        )

    async def _get_customer_history(
        self, customer_email: str, max_orders: int = 50
    ) -> Dict[str, Any]:
        query = f"""
        query {{
            orders(first: {min(max_orders, 250)}, query: "email:{customer_email}", sortKey: CREATED_AT, reverse: true) {{
                edges {{
                    node {{
                        id name createdAt displayFinancialStatus displayFulfillmentStatus
                        totalPriceSet {{ shopMoney {{ amount currencyCode }} }}
                        customer {{
                            id firstName lastName numberOfOrders
                            amountSpent {{ amount currencyCode }}
                        }}
                        lineItems(first: 20) {{ edges {{ node {{ title quantity }} }} }}
                    }}
                }}
            }}
        }}
        """
        result = self._make_graphql_request(query)
        if "error" in result:
            return {"status": "error", "data": None, "message": str(result["error"])}

        orders = result.get("orders", {}).get("edges", [])
        if not orders:
            return {
                "status": "success",
                "data": None,
                "message": f"No orders found for {customer_email}",
            }

        first_customer = orders[0]["node"].get("customer", {})
        customer_info = {
            "email": customer_email,
            "id": first_customer.get("id"),
            "name": f"{first_customer.get('firstName', '')} {first_customer.get('lastName', '')}".strip(),
            "total_orders": first_customer.get("numberOfOrders"),
            "total_spent": first_customer.get("amountSpent", {}).get("amount"),
            "currency": first_customer.get("amountSpent", {}).get("currencyCode"),
        }
        order_list = []
        for oe in orders:
            o = oe["node"]
            order_list.append(
                {
                    "id": o["id"],
                    "name": o["name"],
                    "created_at": o["createdAt"],
                    "financial_status": o.get("displayFinancialStatus"),
                    "fulfillment_status": o.get("displayFulfillmentStatus"),
                    "total": o.get("totalPriceSet", {})
                    .get("shopMoney", {})
                    .get("amount"),
                    "items": [
                        {"title": i["node"]["title"], "quantity": i["node"]["quantity"]}
                        for i in o.get("lineItems", {}).get("edges", [])
                    ],
                }
            )
        return {
            "status": "success",
            "data": {"customer": customer_info, "orders": order_list},
            "message": f"Found {len(order_list)} orders for {customer_email}",
        }


class ShopifyGetInventoryLevels(ShopifyTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="shopify_get_inventory_levels",
            description="Get current inventory levels for all products.",
            inputSchema={
                "type": "object",
                "properties": {"max_results": {"type": "integer", "default": 100}},
            },
            function=self._get_inventory,
        )

    async def _get_inventory(self, max_results: int = 100) -> Dict[str, Any]:
        query = f"""
        query {{
            products(first: {min(max_results, 250)}, query: "status:ACTIVE") {{
                edges {{
                    node {{
                        id title totalInventory tracksInventory
                        variants(first: 50) {{
                            edges {{ node {{ id title sku inventoryQuantity inventoryPolicy }} }}
                        }}
                    }}
                }}
            }}
        }}
        """
        result = self._make_graphql_request(query)
        if "error" in result:
            return {"status": "error", "data": None, "message": str(result["error"])}
        products = []
        for edge in result.get("products", {}).get("edges", []):
            node = edge["node"]
            products.append(
                {
                    "id": node["id"],
                    "title": node["title"],
                    "total_inventory": node.get("totalInventory"),
                    "tracks_inventory": node.get("tracksInventory"),
                    "variants": [
                        {
                            "id": v["node"]["id"],
                            "title": v["node"]["title"],
                            "sku": v["node"].get("sku"),
                            "inventory_quantity": v["node"].get("inventoryQuantity"),
                            "inventory_policy": v["node"].get("inventoryPolicy"),
                        }
                        for v in node.get("variants", {}).get("edges", [])
                    ],
                }
            )
        return {
            "status": "success",
            "data": products,
            "message": f"Inventory for {len(products)} products",
        }


class ShopifyGetLowStockProducts(ShopifyTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="shopify_get_low_stock_products",
            description="Get products that are running low on stock.",
            inputSchema={
                "type": "object",
                "properties": {
                    "threshold": {"type": "integer", "default": 10},
                    "max_results": {"type": "integer", "default": 50},
                },
            },
            function=self._get_low_stock,
        )

    async def _get_low_stock(
        self, threshold: int = 10, max_results: int = 50
    ) -> Dict[str, Any]:
        query = """
        query {
            products(first: 250, query: "status:ACTIVE") {
                edges {
                    node {
                        id title totalInventory
                        variants(first: 50) {
                            edges { node { id title sku inventoryQuantity } }
                        }
                    }
                }
            }
        }
        """
        result = self._make_graphql_request(query)
        if "error" in result:
            return {"status": "error", "data": None, "message": str(result["error"])}

        low_stock = []
        for edge in result.get("products", {}).get("edges", []):
            node = edge["node"]
            total_inv = node.get("totalInventory", 0)
            if total_inv is not None and total_inv <= threshold:
                low_variants = [
                    {
                        "id": v["node"]["id"],
                        "title": v["node"]["title"],
                        "sku": v["node"].get("sku"),
                        "inventory_quantity": v["node"].get("inventoryQuantity"),
                    }
                    for v in node.get("variants", {}).get("edges", [])
                    if v["node"].get("inventoryQuantity") is not None
                    and v["node"]["inventoryQuantity"] <= threshold
                ]
                if low_variants:
                    low_stock.append(
                        {
                            "id": node["id"],
                            "title": node["title"],
                            "total_inventory": total_inv,
                            "low_stock_variants": low_variants,
                        }
                    )
        low_stock.sort(key=lambda x: x["total_inventory"])
        return {
            "status": "success",
            "data": low_stock[:max_results],
            "message": f"Found {len(low_stock)} low stock products",
        }


class ShopifyGetSalesTrends(ShopifyTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="shopify_get_sales_trends",
            description="Get sales trends comparing current period to previous period.",
            inputSchema={
                "type": "object",
                "properties": {
                    "created_after": {"type": "string"},
                    "created_before": {"type": "string"},
                    "compare_previous_period": {"type": "boolean", "default": True},
                },
            },
            function=self._get_trends,
        )

    def _fetch_period(self, start: str, end: str) -> Dict[str, Any]:
        query = f"""
        query {{
            orders(first: 250, query: "created_at:>={start} AND created_at:<{end} AND financial_status:paid") {{
                edges {{
                    node {{
                        totalPriceSet {{ shopMoney {{ amount currencyCode }} }}
                        lineItems(first: 100) {{ edges {{ node {{ quantity }} }} }}
                    }}
                }}
            }}
        }}
        """
        result = self._make_graphql_request(query)
        if "error" in result:
            return {"error": result["error"]}
        orders = result.get("orders", {}).get("edges", [])
        total_rev = sum(
            float(
                o["node"].get("totalPriceSet", {}).get("shopMoney", {}).get("amount", 0)
            )
            for o in orders
        )
        total_items = sum(
            sum(
                i["node"]["quantity"]
                for i in o["node"].get("lineItems", {}).get("edges", [])
            )
            for o in orders
        )
        cur = (
            orders[0]["node"]
            .get("totalPriceSet", {})
            .get("shopMoney", {})
            .get("currencyCode")
            if orders
            else None
        )
        return {
            "total_orders": len(orders),
            "total_revenue": round(total_rev, 2),
            "total_items_sold": total_items,
            "average_order_value": round(total_rev / len(orders), 2) if orders else 0,
            "currency": cur,
        }

    async def _get_trends(
        self,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        compare_previous_period: bool = True,
    ) -> Dict[str, Any]:
        now = datetime.now()
        end_dt = (
            datetime.strptime(created_before, "%Y-%m-%d") if created_before else now
        )
        start_dt = (
            datetime.strptime(created_after, "%Y-%m-%d")
            if created_after
            else end_dt - timedelta(days=30)
        )

        current = self._fetch_period(
            start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
        )
        if "error" in current:
            return {"status": "error", "data": None, "message": str(current["error"])}

        data: Dict[str, Any] = {
            "current_period": {
                "start": start_dt.strftime("%Y-%m-%d"),
                "end": end_dt.strftime("%Y-%m-%d"),
                **current,
            }
        }

        if compare_previous_period:
            days = (end_dt - start_dt).days
            prev_end = start_dt - timedelta(days=1)
            prev_start = prev_end - timedelta(days=days)
            previous = self._fetch_period(
                prev_start.strftime("%Y-%m-%d"), prev_end.strftime("%Y-%m-%d")
            )
            if "error" not in previous:
                data["previous_period"] = {
                    "start": prev_start.strftime("%Y-%m-%d"),
                    "end": prev_end.strftime("%Y-%m-%d"),
                    **previous,
                }
                rev_change = (
                    (
                        (current["total_revenue"] - previous["total_revenue"])
                        / previous["total_revenue"]
                        * 100
                    )
                    if previous["total_revenue"] > 0
                    else (100 if current["total_revenue"] > 0 else 0)
                )
                ord_change = (
                    (
                        (current["total_orders"] - previous["total_orders"])
                        / previous["total_orders"]
                        * 100
                    )
                    if previous["total_orders"] > 0
                    else (100 if current["total_orders"] > 0 else 0)
                )
                data["comparison"] = {
                    "revenue_change_percent": round(rev_change, 2),
                    "orders_change_percent": round(ord_change, 2),
                    "revenue_trend": "up"
                    if rev_change > 0
                    else ("down" if rev_change < 0 else "flat"),
                    "orders_trend": "up"
                    if ord_change > 0
                    else ("down" if ord_change < 0 else "flat"),
                }
        return {"status": "success", "data": data, "message": "Sales trends retrieved"}


class ShopifyGetAverageOrderValue(ShopifyTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="shopify_get_average_order_value",
            description="Get average order value over time.",
            inputSchema={
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": "string",
                        "enum": ["day", "week", "month"],
                        "default": "day",
                    },
                    "created_after": {"type": "string"},
                    "created_before": {"type": "string"},
                },
            },
            function=self._get_aov,
        )

    async def _get_aov(
        self,
        group_by: str = "day",
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
    ) -> Dict[str, Any]:
        query_parts = ["financial_status:paid"]
        if created_after:
            query_parts.append(f"created_at:>={created_after}")
        if created_before:
            query_parts.append(f"created_at:<={created_before}")
        query_filter = " AND ".join(query_parts)

        query = f"""
        query {{
            orders(first: 250, query: "{query_filter}", sortKey: CREATED_AT) {{
                edges {{
                    node {{
                        createdAt
                        totalPriceSet {{ shopMoney {{ amount currencyCode }} }}
                    }}
                }}
            }}
        }}
        """
        result = self._make_graphql_request(query)
        if "error" in result:
            return {"status": "error", "data": None, "message": str(result["error"])}

        orders = result.get("orders", {}).get("edges", [])
        if not orders:
            return {"status": "success", "data": None, "message": "No orders found"}

        grouped: Dict[str, List[float]] = {}
        currency = None
        for oe in orders:
            o = oe["node"]
            created_at = o["createdAt"][:10]
            amount = float(
                o.get("totalPriceSet", {}).get("shopMoney", {}).get("amount", 0)
            )
            currency = (
                o.get("totalPriceSet", {}).get("shopMoney", {}).get("currencyCode")
            )
            if group_by == "week":
                dt = datetime.strptime(created_at, "%Y-%m-%d")
                key = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
            elif group_by == "month":
                key = created_at[:7]
            else:
                key = created_at
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(amount)

        breakdown = [
            {
                "period": k,
                "order_count": len(v),
                "total_revenue": round(sum(v), 2),
                "average_order_value": round(sum(v) / len(v), 2),
            }
            for k, v in sorted(grouped.items())
        ]
        overall = sum(d["total_revenue"] for d in breakdown) / sum(
            d["order_count"] for d in breakdown
        )
        data = {
            "group_by": group_by,
            "overall_average_order_value": round(overall, 2),
            "currency": currency,
            "breakdown": breakdown,
        }
        return {"status": "success", "data": data, "message": "AOV data retrieved"}


class ShopifyGetRepeatCustomers(ShopifyTools):
    def get_tool(self) -> Tool:
        return Tool(
            name="shopify_get_repeat_customers",
            description="Find customers who have made multiple purchases.",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_orders": {"type": "integer", "default": 2},
                    "limit": {"type": "integer", "default": 50},
                    "created_after": {"type": "string"},
                    "created_before": {"type": "string"},
                },
            },
            function=self._get_repeat_customers,
        )

    async def _get_repeat_customers(
        self,
        min_orders: int = 2,
        limit: int = 50,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
    ) -> Dict[str, Any]:
        query_parts = ["financial_status:paid"]
        if created_after:
            query_parts.append(f"created_at:>={created_after}")
        if created_before:
            query_parts.append(f"created_at:<={created_before}")
        query_filter = " AND ".join(query_parts)

        query = f"""
        query {{
            orders(first: 250, query: "{query_filter}") {{
                edges {{
                    node {{
                        customer {{
                            id email firstName lastName numberOfOrders
                            amountSpent {{ amount currencyCode }}
                        }}
                        totalPriceSet {{ shopMoney {{ amount }} }}
                    }}
                }}
            }}
        }}
        """
        result = self._make_graphql_request(query)
        if "error" in result:
            return {"status": "error", "data": None, "message": str(result["error"])}

        customer_data: Dict[str, Dict[str, Any]] = {}
        for oe in result.get("orders", {}).get("edges", []):
            customer = oe["node"].get("customer")
            if not customer or not customer.get("id"):
                continue
            cid = customer["id"]
            amount = float(
                oe["node"]
                .get("totalPriceSet", {})
                .get("shopMoney", {})
                .get("amount", 0)
            )
            if cid not in customer_data:
                customer_data[cid] = {
                    "id": cid,
                    "email": customer.get("email"),
                    "name": f"{customer.get('firstName', '')} {customer.get('lastName', '')}".strip(),
                    "total_orders_all_time": customer.get("numberOfOrders"),
                    "total_spent_all_time": customer.get("amountSpent", {}).get(
                        "amount"
                    ),
                    "currency": customer.get("amountSpent", {}).get("currencyCode"),
                    "orders_in_period": 0,
                    "spent_in_period": 0.0,
                }
            customer_data[cid]["orders_in_period"] += 1
            customer_data[cid]["spent_in_period"] += amount

        repeat = [
            {**c, "spent_in_period": round(c["spent_in_period"], 2)}
            for c in customer_data.values()
            if c["orders_in_period"] >= min_orders
        ]
        repeat.sort(key=lambda x: x["orders_in_period"], reverse=True)

        data = {
            "min_orders_threshold": min_orders,
            "repeat_customer_count": len(repeat),
            "customers": repeat[:limit],
        }
        return {
            "status": "success",
            "data": data,
            "message": f"Found {len(repeat)} repeat customers",
        }
