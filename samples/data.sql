WITH RECURSIVE seq(n) AS (
    SELECT 1
    UNION ALL
    SELECT n + 1 FROM seq WHERE n < 2000
)
INSERT INTO customers(id, name, city, segment, created_at)
SELECT
    n,
    'Customer ' || printf('%05d', n),
    CASE (n * 7 + 157) % 6
        WHEN 0 THEN 'Shanghai'
        WHEN 1 THEN 'Beijing'
        WHEN 2 THEN 'Shenzhen'
        WHEN 3 THEN 'Hangzhou'
        WHEN 4 THEN 'Chengdu'
        ELSE 'Wuhan'
    END,
    CASE (n * 5 + 157) % 3
        WHEN 0 THEN 'retail'
        WHEN 1 THEN 'business'
        ELSE 'partner'
    END,
    date('2023-01-01', printf('+%d days', (n * 13) % 365))
FROM seq;

WITH RECURSIVE seq(n) AS (
    SELECT 1
    UNION ALL
    SELECT n + 1 FROM seq WHERE n < 20000
)
INSERT INTO orders(id, customer_id, status, amount, created_at)
SELECT
    n,
    ((n * 37 + 157) % 2000) + 1,
    CASE (n * 11 + 157) % 4
        WHEN 0 THEN 'paid'
        WHEN 1 THEN 'pending'
        WHEN 2 THEN 'shipped'
        ELSE 'cancelled'
    END,
    round((((n * 7919 + 157) % 200000) / 100.0) + 1.0, 2),
    date('2023-01-01', printf('+%d days', (n * 17) % 700))
FROM seq;
