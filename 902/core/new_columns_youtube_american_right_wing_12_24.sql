-- ------------------------------------------------------------------------------
-- 1) Pourcentage de réponses
-- ------------------------------------------------------------------------------
-- Calcule, pour chaque vidéo, la proportion (%) de commentaires
-- qui répondent à un autre commentaire (champ parent non NULL).
-- ------------------------------------------------------------------------------
UPDATE video v
SET video_reply_percentage = sub.reply_percentage
FROM (
    SELECT
        comment_video_id AS vid,
        100.0 * SUM(CASE WHEN comment_parent_comment_id IS NOT NULL THEN 1 ELSE 0 END)::numeric
               / COUNT(*) AS reply_percentage
    FROM comment
    GROUP BY comment_video_id
) AS sub
WHERE v.video_id = sub.vid;


-- ------------------------------------------------------------------------------
-- 2) Longueur moyenne des commentaires
-- ------------------------------------------------------------------------------
-- Calcule la longueur (en caractères) de chaque commentaire
-- puis en fait la moyenne par vidéo.
-- ------------------------------------------------------------------------------
UPDATE video v
SET video_avg_comment_length = sub.avg_length
FROM (
    SELECT
        comment_video_id AS vid,
        AVG(LENGTH(comment_text_display)) AS avg_length
    FROM comment
    GROUP BY comment_video_id
) AS sub
WHERE v.video_id = sub.vid;


-- ------------------------------------------------------------------------------
-- 3) Diversité des auteurs
-- ------------------------------------------------------------------------------
-- Ratio = (nombre d'auteurs distincts) / (nombre total de commentaires),
-- pour mesurer la dispersion ou la concentration des contributions.
-- ------------------------------------------------------------------------------
UPDATE video v
SET video_author_diversity = sub.author_div
FROM (
    SELECT
      comment_video_id AS vid,
      COUNT(*) AS total_comments,
      COUNT(DISTINCT comment_author_channel_id) AS nb_authors,
      COUNT(DISTINCT comment_author_channel_id)::float / COUNT(*) AS author_div
    FROM comment
    GROUP BY comment_video_id
) AS sub
WHERE v.video_id = sub.vid;


-- ------------------------------------------------------------------------------
-- 4) Profondeur des fils (max et moyenne)
-- ------------------------------------------------------------------------------
-- Utilise une CTE récursive pour calculer la profondeur de chaque commentaire
-- en remontant ses parents. Puis agrège par vidéo (max et moyenne).
-- ------------------------------------------------------------------------------
WITH RECURSIVE thread_depth AS (
    SELECT
      comment_id,
      comment_parent_comment_id,
      comment_video_id,
      1 AS depth
    FROM comment

    UNION ALL

    SELECT
      c.comment_id,
      c.comment_parent_comment_id,
      c.comment_video_id,
      td.depth + 1 AS depth
    FROM comment c
    JOIN thread_depth td 
         ON c.comment_parent_comment_id = td.comment_id
),
depth_agg AS (
    SELECT
      comment_video_id AS vid,
      MAX(depth) AS max_depth,
      AVG(depth) AS avg_depth
    FROM thread_depth
    GROUP BY comment_video_id
)
UPDATE video v
SET video_max_thread_depth = depth_agg.max_depth,
    video_avg_thread_depth = depth_agg.avg_depth
FROM depth_agg
WHERE v.video_id = depth_agg.vid;


-- ------------------------------------------------------------------------------
-- 5) Délai moyen de publication (en heures)
-- ------------------------------------------------------------------------------
-- Mesure le délai moyen entre la publication de la vidéo et la publication
-- de chaque commentaire.
-- ------------------------------------------------------------------------------
UPDATE video v
SET video_avg_delay_hours = sub.avg_delay
FROM (
    SELECT
      c.comment_video_id AS vid,
      AVG(EXTRACT(EPOCH FROM (c.comment_published_at - v.video_published_at))) / 3600.0
         AS avg_delay
    FROM comment c
    JOIN video v ON v.video_id = c.comment_video_id
    GROUP BY c.comment_video_id
) AS sub
WHERE v.video_id = sub.vid;


-- ------------------------------------------------------------------------------
-- 6) Ratio "likes / nombre de commentaires" (côté vidéo)
-- ------------------------------------------------------------------------------
-- Suppose que video.video_like_count et video.video_comment_count sont déjà
-- dans la table 'video'. On crée ou met à jour un ratio simple.
-- ------------------------------------------------------------------------------
UPDATE video
SET video_like_comment_ratio = CASE
    WHEN video_comment_count = 0 THEN NULL
    ELSE (video_like_count::float / video_comment_count)
END;


-- ------------------------------------------------------------------------------
-- 7) Distribution des likes sur les commentaires : moyenne + % "populaires"
-- ------------------------------------------------------------------------------
-- Calcule la moyenne de likes par commentaire et la proportion de commentaires
-- qui dépassent un certain seuil (ex: 10).
-- ------------------------------------------------------------------------------
UPDATE video v
SET video_avg_comment_likes      = sub.avg_likes,
    video_ratio_popular_comments = sub.popular_ratio
FROM (
    SELECT
        comment_video_id AS vid,
        AVG(comment_like_count) AS avg_likes,
        100.0 * SUM(CASE WHEN comment_like_count >= 10 THEN 1 ELSE 0 END)::float 
               / COUNT(*) AS popular_ratio
    FROM comment
    GROUP BY comment_video_id
) sub
WHERE v.video_id = sub.vid;


-- ------------------------------------------------------------------------------
-- 8) Richesse lexicale (simple)
-- ------------------------------------------------------------------------------
-- Concatène tous les commentaires d'une vidéo, sépare par espaces,
-- et calcule la proportion (mots distincts / total de mots).
-- ------------------------------------------------------------------------------
WITH all_text AS (
    SELECT
        c.comment_video_id AS vid,
        lower(string_agg(c.comment_text_display, ' ')) AS full_text
    FROM comment c
    GROUP BY c.comment_video_id
),
lexical_calc AS (
    SELECT
        vid,
        cardinality(string_to_array(full_text, ' '))::float AS total_words,
        cardinality(ARRAY(
           SELECT DISTINCT unnest(string_to_array(full_text, ' '))
        ))::float AS distinct_words
    FROM all_text
)
UPDATE video v
SET video_lexical_richness = CASE
   WHEN lc.total_words > 0 THEN (lc.distinct_words / lc.total_words)
   ELSE 0
END
FROM lexical_calc lc
WHERE v.video_id = lc.vid;


-- ------------------------------------------------------------------------------
-- 9) Ponctuation et emphase : nb moyen de "!" par commentaire
-- ------------------------------------------------------------------------------
-- On remplace tout ce qui n'est pas un "!" par rien, puis on mesure la longueur
-- de la chaîne qui reste. On en fait la moyenne.
-- ------------------------------------------------------------------------------
UPDATE video v
SET video_avg_exclamations = sub.avg_ex
FROM (
    SELECT
      comment_video_id AS vid,
      AVG(LENGTH(regexp_replace(comment_text_display, '[^!]', '', 'g'))) AS avg_ex
    FROM comment
    GROUP BY comment_video_id
) AS sub
WHERE v.video_id = sub.vid;


-- ------------------------------------------------------------------------------
-- 10) Ponctuation et emphase : proportion de commentaires en MAJUSCULES
-- ------------------------------------------------------------------------------
-- Compare la version normale du texte à sa version UPPER(). 
-- S'il y a égalité, on considère que le commentaire est entièrement en maj.
-- ------------------------------------------------------------------------------
UPDATE video v
SET video_ratio_all_caps = sub.ratio_all_caps
FROM (
    SELECT
      comment_video_id AS vid,
      100.0 * SUM(
          CASE WHEN comment_text_display = UPPER(comment_text_display)
               THEN 1 ELSE 0 END
      )::float / COUNT(*) AS ratio_all_caps
    FROM comment
    GROUP BY comment_video_id
) AS sub
WHERE v.video_id = sub.vid;


-- ------------------------------------------------------------------------------
-- 11) Coefficient de variation des délais (en heures)
-- ------------------------------------------------------------------------------
-- Mesure la dispersion temporelle des commentaires : 
-- CV = (écart-type des délais) / (moyenne des délais).
-- Un CV élevé indique des délais très étalés dans le temps.
-- ------------------------------------------------------------------------------
WITH stats AS (
    SELECT
        c.comment_video_id AS vid,
        -- Moyenne des délais en heures
        AVG(EXTRACT(EPOCH FROM (c.comment_published_at - v.video_published_at)) / 3600.0) 
            AS mean_delay_hours,
        -- Écart-type (population) des délais en heures
        STDDEV_POP(EXTRACT(EPOCH FROM (c.comment_published_at - v.video_published_at)) / 3600.0) 
            AS std_delay_hours
    FROM comment c
    JOIN video v ON v.video_id = c.comment_video_id
    GROUP BY c.comment_video_id
)
UPDATE video v
SET video_delay_cv = CASE 
    WHEN stats.mean_delay_hours = 0 THEN NULL
    ELSE (stats.std_delay_hours / stats.mean_delay_hours)
END
FROM stats
WHERE v.video_id = stats.vid;


-- ------------------------------------------------------------------------------
-- Met à jour sentiment_star en extrayant le chiffre du label "<chiffre> star(s)?"
-- grâce à une regex. Si le format n’est pas respecté, on met la valeur à NULL.
-- Ex. "3 stars" -> 3, "3 star" -> 3, "3s" -> NULL.
-- ------------------------------------------------------------------------------
UPDATE sentiment
SET sentiment_star = CASE
    WHEN (sentiment->0->>'label') ~ '^\d+ star(s)?$'
         THEN substring((sentiment->0->>'label') from '^(\d+)')::int
    ELSE NULL
END;


-- ------------------------------------------------------------------------------
-- 12) video_avg_sentiment
-- ------------------------------------------------------------------------------
-- Calcule la moyenne du score (1 à 5) pour chaque vidéo,
-- à partir de la colonne sentiment_star dans la table `sentiment`.
-- Puis met à jour la colonne video_avg_sentiment dans `video`.
-- ------------------------------------------------------------------------------
UPDATE video
SET video_avg_sentiment = sub.avg_sentiment
FROM (
    SELECT
        sentiment_video_id AS vid,
        AVG(sentiment_star) AS avg_sentiment
    FROM sentiment
    GROUP BY sentiment_video_id
) AS sub
WHERE video.video_id = sub.vid;


-- ------------------------------------------------------------------------------
-- 13) video_sentiment_stddev
-- ------------------------------------------------------------------------------
-- Calcule l'écart-type du score (1 à 5) pour chaque vidéo,
-- en se basant sur la colonne sentiment_star,
-- et met à jour la colonne video_sentiment_stddev.
-- ------------------------------------------------------------------------------
UPDATE video
SET video_sentiment_stddev = sub.stddev_sentiment
FROM (
    SELECT
        sentiment_video_id AS vid,
        STDDEV(sentiment_star) AS stddev_sentiment
    FROM sentiment
    GROUP BY sentiment_video_id
) AS sub
WHERE video.video_id = sub.vid;


-- ------------------------------------------------------------------------------
-- 14) video_sentiment_polarity_ratio
-- ------------------------------------------------------------------------------
-- Calcule la différence (#positifs - #négatifs) / #total.
-- - "positif"  = sentiment_star >= 4
-- - "négatif"  = sentiment_star <= 2
-- - "neutre"   = 3 (non compté).
-- ------------------------------------------------------------------------------
UPDATE video
SET video_sentiment_polarity_ratio = sub.polarity_ratio
FROM (
    WITH counts AS (
        SELECT
            sentiment_video_id AS vid,
            SUM(
                CASE WHEN sentiment_star >= 4 THEN 1 ELSE 0 END
            ) AS cnt_pos,
            SUM(
                CASE WHEN sentiment_star <= 2 THEN 1 ELSE 0 END
            ) AS cnt_neg,
            COUNT(*) AS cnt_total
        FROM sentiment
        GROUP BY sentiment_video_id
    )
    SELECT
        vid,
        ( (cnt_pos - cnt_neg)::numeric / NULLIF(cnt_total, 0) ) AS polarity_ratio
    FROM counts
) AS sub
WHERE video.video_id = sub.vid;


-- ------------------------------------------------------------------------------
-- 15) video_strong_emotion_ratio
-- ------------------------------------------------------------------------------
-- Calcule la proportion de segments "extrêmes" (sentiment_star = 1 ou 5)
-- par rapport au nombre total de segments.
-- ------------------------------------------------------------------------------
UPDATE video
SET video_strong_emotion_ratio = sub.strong_ratio
FROM (
    SELECT
        sentiment_video_id AS vid,
        (
            SUM(
                CASE WHEN sentiment_star IN (1, 5) THEN 1 ELSE 0 END
            )::numeric
            / COUNT(*)
        ) AS strong_ratio
    FROM sentiment
    GROUP BY sentiment_video_id
) AS sub
WHERE video.video_id = sub.vid;


-- ------------------------------------------------------------------------------
-- 16) video_sentiment_trend
-- ------------------------------------------------------------------------------
-- Évalue la "pente" (trend) du sentiment sur la vidéo :
--   - On numérote les segments (i) par ROW_NUMBER() selon sentiment_start_token_idx.
--   - On calcule la pente d'une régression linéaire simple : s = a + b*i.
-- ------------------------------------------------------------------------------
UPDATE video
SET video_sentiment_trend = sub.slope
FROM (
    WITH ranked AS (
        SELECT
            sentiment_video_id AS vid,
            sentiment_star::numeric AS s,
            ROW_NUMBER() OVER (
                PARTITION BY sentiment_video_id
                ORDER BY sentiment_start_token_idx
            ) AS i
        FROM sentiment
    ),
    stats AS (
        SELECT
            vid,
            COUNT(*)::numeric AS n,
            SUM(i) AS sum_i,
            SUM(s) AS sum_s,
            SUM(i * s) AS sum_is,
            SUM(i * i) AS sum_i2
        FROM ranked
        GROUP BY vid
    )
    SELECT
        vid,
        CASE 
            WHEN (sum_i2 - (sum_i^2 / n)) = 0 THEN 0
            ELSE (
                ( sum_is - ( sum_i * sum_s / n ) )
                /
                ( sum_i2 - ( sum_i^2 / n ) )
            )
        END AS slope
    FROM stats
) AS sub
WHERE video.video_id = sub.vid;


-- ------------------------------------------------------------------------------
-- 17) video_sentiment_sign_changes_prop
-- ------------------------------------------------------------------------------
-- Calcule le nombre de changements de signe (positif <-> négatif),
-- en se basant sur sentiment_star :
--   - "positif" = sentiment_star >= 4 => +1
--   - "négatif" = sentiment_star <= 2 => -1
--   - "neutre"  = 3 => 0
-- Divise ensuite ce nombre par le total de segments pour un ratio.
-- ------------------------------------------------------------------------------
WITH cte AS (
    SELECT
        sentiment_video_id AS vid,
        CASE
            WHEN sentiment_star >= 4 THEN  1
            WHEN sentiment_star <= 2 THEN -1
            ELSE 0
        END AS s,
        LAG(
            CASE
                WHEN sentiment_star >= 4 THEN  1
                WHEN sentiment_star <= 2 THEN -1
                ELSE 0
            END
        ) OVER (
          PARTITION BY sentiment_video_id
          ORDER BY sentiment_start_token_idx
        ) AS s_prev
    FROM sentiment
),
sign_changes AS (
    SELECT
        vid,
        SUM(
            CASE
                WHEN s_prev IS NULL THEN 0
                WHEN s_prev != s AND s != 0 AND s_prev != 0 THEN 1
                ELSE 0
            END
        ) AS nb_sign_changes,
        COUNT(*) AS nb_segments
    FROM cte
    GROUP BY vid
)
UPDATE video
SET video_sentiment_sign_changes_prop = sub.sign_change_ratio
FROM (
    SELECT
        vid,
        ( nb_sign_changes::numeric / NULLIF(nb_segments, 0) ) AS sign_change_ratio
    FROM sign_changes
) AS sub
WHERE video.video_id = sub.vid;




/*
Ce script met à jour les statistiques agrégées des chaînes en se basant sur les données calculées pour les vidéos associées. 
Il calcule et applique des moyennes pour divers indicateurs liés à l'engagement, au contenu et au sentiment.
*/

UPDATE channel c
SET
    channel_avg_reply_percentage         = sub.avg_reply_percentage,
    channel_avg_comment_length           = sub.avg_comment_length,
    channel_avg_author_diversity         = sub.avg_author_diversity,
    channel_avg_max_thread_depth         = sub.avg_max_thread_depth,
    channel_avg_thread_depth             = sub.avg_thread_depth,
    channel_avg_delay_hours              = sub.avg_delay_hours,
    channel_like_comment_ratio           = sub.like_comment_ratio,
    channel_avg_comment_likes            = sub.avg_comment_likes,
    channel_ratio_popular_comments       = sub.ratio_popular_comments,
    channel_lexical_richness            = sub.lexical_richness,
    channel_avg_exclamations            = sub.avg_exclamations,
    channel_ratio_all_caps              = sub.ratio_all_caps,
    channel_delay_cv                    = sub.delay_cv,
    channel_avg_sentiment               = sub.avg_sentiment,
    channel_sentiment_stddev            = sub.sentiment_stddev,
    channel_sentiment_polarity_ratio    = sub.sentiment_polarity_ratio,
    channel_strong_emotion_ratio        = sub.strong_emotion_ratio,
    channel_sentiment_trend             = sub.sentiment_trend,
    channel_sentiment_sign_changes_prop = sub.sentiment_sign_changes_prop
FROM (
    SELECT
        v.video_channel_id AS chan_id,
        
        /* Moyenne sur toutes les vidéos de la chaîne (attention: NULLs si pas de vidéos) */
        AVG(video_reply_percentage)         AS avg_reply_percentage,
        AVG(video_avg_comment_length)       AS avg_comment_length,
        AVG(video_author_diversity)         AS avg_author_diversity,
        AVG(video_max_thread_depth)::numeric(10,2) AS avg_max_thread_depth,  -- on cast en numeric(10,2)
        AVG(video_avg_thread_depth)         AS avg_thread_depth,
        AVG(video_avg_delay_hours)          AS avg_delay_hours,
        AVG(video_like_comment_ratio)       AS like_comment_ratio,
        AVG(video_avg_comment_likes)        AS avg_comment_likes,
        AVG(video_ratio_popular_comments)   AS ratio_popular_comments,
        AVG(video_lexical_richness)         AS lexical_richness,
        AVG(video_avg_exclamations)         AS avg_exclamations,
        AVG(video_ratio_all_caps)           AS ratio_all_caps,
        AVG(video_delay_cv)                 AS delay_cv,
        AVG(video_avg_sentiment)            AS avg_sentiment,
        AVG(video_sentiment_stddev)         AS sentiment_stddev,
        AVG(video_sentiment_polarity_ratio) AS sentiment_polarity_ratio,
        AVG(video_strong_emotion_ratio)     AS strong_emotion_ratio,
        AVG(video_sentiment_trend)          AS sentiment_trend,
        AVG(video_sentiment_sign_changes_prop) AS sentiment_sign_changes_prop
        
    FROM video v
    GROUP BY v.video_channel_id
) AS sub
WHERE c.channel_id = sub.chan_id;

