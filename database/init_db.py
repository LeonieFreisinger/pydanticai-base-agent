"""
Database initialization script for the SQL Agent Showcase.

Creates an SQLite database with realistic e-commerce data including:
- 10 suppliers
- 50 products with inventory data
- 30 customers
- 200 orders with seasonal patterns
- 500+ order items

Run: python database/init_db.py
"""

import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# Seed for reproducibility
random.seed(42)

DB_PATH = Path(__file__).parent.parent / "showcase.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def create_database() -> None:
    """Create the database schema."""
    # Remove existing database
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)

    # Load and execute schema
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())

    conn.commit()
    conn.close()
    print(f"✓ Created database schema at {DB_PATH}")


def seed_suppliers(conn: sqlite3.Connection) -> list[int]:
    """Seed supplier data."""
    suppliers = [
        ("TechParts Inc.", "orders@techparts.com", "USA", 5, 0.98),
        ("Global Electronics", "supply@globalelec.com", "China", 14, 0.92),
        ("EuroComponents", "info@eurocomp.eu", "Germany", 7, 0.96),
        ("FastShip Logistics", "orders@fastship.com", "USA", 3, 0.94),
        ("Asian Manufacturing Co.", "sales@asianmfg.cn", "China", 21, 0.88),
        ("Nordic Supplies", "hello@nordicsupplies.se", "Sweden", 10, 0.97),
        ("MexiParts", "ventas@mexiparts.mx", "Mexico", 8, 0.91),
        ("UK Electronics Ltd", "orders@ukelec.co.uk", "UK", 6, 0.95),
        ("Canadian Components", "info@cancomp.ca", "Canada", 4, 0.93),
        ("Australian Tech", "sales@austech.au", "Australia", 12, 0.90),
    ]

    cursor = conn.cursor()
    cursor.executemany(
        """INSERT INTO suppliers (name, contact_email, country, lead_time_days, reliability_score)
           VALUES (?, ?, ?, ?, ?)""",
        suppliers,
    )
    conn.commit()

    # Get IDs
    cursor.execute("SELECT id FROM suppliers")
    supplier_ids = [row[0] for row in cursor.fetchall()]
    print(f"✓ Seeded {len(suppliers)} suppliers")
    return supplier_ids


def seed_products(conn: sqlite3.Connection, supplier_ids: list[int]) -> list[int]:
    """Seed product data with realistic inventory patterns."""
    categories = ["Electronics", "Components", "Accessories", "Tools", "Cables"]

    products = []
    for i in range(1, 51):
        category = random.choice(categories)
        base_price = random.uniform(10, 500)

        # Create realistic inventory scenarios
        if i % 10 == 0:  # 10% are low stock (interesting for queries)
            stock = random.randint(0, 5)
            reorder = random.randint(10, 20)
        elif i % 7 == 0:  # Some are overstocked
            stock = random.randint(200, 500)
            reorder = random.randint(20, 50)
        else:  # Normal stock
            stock = random.randint(20, 150)
            reorder = random.randint(10, 30)

        products.append(
            (
                f"SKU-{i:04d}",
                f"{category} Product {i}",
                category,
                round(base_price, 2),
                round(base_price * 0.6, 2),  # 40% margin
                stock,
                reorder,
                max(5, reorder // 2),  # safety stock
                random.choice(supplier_ids),
                True,
            )
        )

    cursor = conn.cursor()
    cursor.executemany(
        """INSERT INTO products (sku, name, category, unit_price, cost_price, 
           stock_level, reorder_point, safety_stock, supplier_id, is_active)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        products,
    )
    conn.commit()

    cursor.execute("SELECT id FROM products")
    product_ids = [row[0] for row in cursor.fetchall()]
    print(f"✓ Seeded {len(products)} products")
    return product_ids


def seed_customers(conn: sqlite3.Connection) -> list[int]:
    """Seed customer data."""
    customer_types = ["regular", "premium", "enterprise"]
    countries = ["USA", "Germany", "UK", "France", "Canada", "Australia", "Japan"]

    customers = []
    for i in range(1, 31):
        ctype = random.choices(customer_types, weights=[0.6, 0.3, 0.1])[0]
        customers.append(
            (
                f"Customer {i}",
                f"customer{i}@example.com",
                f"Company {i}" if random.random() > 0.3 else None,
                random.choice(countries),
                ctype,
            )
        )

    cursor = conn.cursor()
    cursor.executemany(
        """INSERT INTO customers (name, email, company, country, customer_type)
           VALUES (?, ?, ?, ?, ?)""",
        customers,
    )
    conn.commit()

    cursor.execute("SELECT id FROM customers")
    customer_ids = [row[0] for row in cursor.fetchall()]
    print(f"✓ Seeded {len(customers)} customers")
    return customer_ids


def seed_orders(conn: sqlite3.Connection, customer_ids: list[int], product_ids: list[int]) -> None:
    """Seed orders with seasonal patterns and realistic distributions."""
    cursor = conn.cursor()

    statuses = ["pending", "confirmed", "shipped", "delivered", "cancelled"]
    status_weights = [0.1, 0.15, 0.2, 0.5, 0.05]
    payment_methods = ["credit_card", "bank_transfer", "paypal"]

    # Generate orders over the past year with seasonal patterns
    base_date = datetime.now() - timedelta(days=365)
    orders_data = []
    order_items_data = []
    order_id = 1

    for day_offset in range(365):
        current_date = base_date + timedelta(days=day_offset)
        month = current_date.month

        # Seasonal multiplier (more orders in Q4)
        if month in [11, 12]:  # Holiday season
            daily_orders = random.randint(1, 4)
        elif month in [1, 2]:  # Post-holiday slump
            daily_orders = random.randint(0, 1)
        else:
            daily_orders = random.randint(0, 2)

        for _ in range(daily_orders):
            customer_id = random.choice(customer_ids)
            status = random.choices(statuses, weights=status_weights)[0]

            # Generate order items
            num_items = random.randint(1, 5)
            selected_products = random.sample(product_ids, min(num_items, len(product_ids)))

            order_total = 0.0
            for product_id in selected_products:
                cursor.execute("SELECT unit_price FROM products WHERE id = ?", (product_id,))
                unit_price = cursor.fetchone()[0]
                quantity = random.randint(1, 10)
                discount = random.choice([0, 0, 0, 5, 10, 15])  # Most have no discount

                item_total = unit_price * quantity * (1 - discount / 100)
                order_total += item_total

                order_items_data.append((order_id, product_id, quantity, unit_price, discount))

            cursor.execute("SELECT country FROM customers WHERE id = ?", (customer_id,))
            shipping_country = cursor.fetchone()[0]

            orders_data.append(
                (
                    customer_id,
                    current_date.isoformat(),
                    status,
                    round(order_total, 2),
                    shipping_country,
                    random.choice(payment_methods),
                )
            )

            order_id += 1

    # Insert orders
    cursor.executemany(
        """INSERT INTO orders (customer_id, order_date, status, total_amount, 
           shipping_country, payment_method)
           VALUES (?, ?, ?, ?, ?, ?)""",
        orders_data,
    )

    # Insert order items
    cursor.executemany(
        """INSERT INTO order_items (order_id, product_id, quantity, unit_price, discount_percent)
           VALUES (?, ?, ?, ?, ?)""",
        order_items_data,
    )

    # Update customer totals
    cursor.execute("""
        UPDATE customers SET 
            total_orders = (SELECT COUNT(*) FROM orders WHERE orders.customer_id = customers.id),
            total_spent = (SELECT COALESCE(SUM(total_amount), 0) FROM orders WHERE orders.customer_id = customers.id)
    """)

    conn.commit()
    print(f"✓ Seeded {len(orders_data)} orders with {len(order_items_data)} order items")


def main() -> None:
    """Initialize the showcase database with seed data."""
    print("\n🚀 Initializing SQL Agent Showcase Database\n")

    create_database()

    conn = sqlite3.connect(DB_PATH)

    supplier_ids = seed_suppliers(conn)
    product_ids = seed_products(conn, supplier_ids)
    customer_ids = seed_customers(conn)
    seed_orders(conn, customer_ids, product_ids)

    conn.close()

    print(f"\n✅ Database ready at: {DB_PATH}")
    print("   Run the agent with: make run\n")


if __name__ == "__main__":
    main()
