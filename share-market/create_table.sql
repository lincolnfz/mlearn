CREATE TABLE `daily_price` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `symbol_id` varchar(32) NOT NULL,
  `price_date` datetime NOT NULL,
  `created_date` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `last_updated_date` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `open_price` decimal(19,4) DEFAULT NULL,
  `high_price` decimal(19,4) DEFAULT NULL,
  `close_price` decimal(19,4) DEFAULT NULL,
  `low_price` decimal(19,4) DEFAULT NULL,
  `adj_close_price` decimal(19,4) DEFAULT NULL,
  `volume` decimal(19,4) DEFAULT NULL,
  `daily_pricecol` varchar(45) DEFAULT NULL,
  `price_change` decimal(19,4) DEFAULT NULL,
  `p_change` decimal(19,4) DEFAULT NULL,
  `ma5` decimal(19,4) DEFAULT NULL,
  `ma10` decimal(19,4) DEFAULT NULL,
  `ma20` decimal(19,4) DEFAULT NULL,
  `v_ma5` decimal(19,4) DEFAULT NULL,
  `v_ma10` decimal(19,4) DEFAULT NULL,
  `v_ma20` decimal(19,4) DEFAULT NULL,
  `turnover` varchar(45) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `index_symbol_id` (`symbol_id`)
) ENGINE=InnoDB AUTO_INCREMENT=0 DEFAULT CHARSET=utf8;

CREATE TABLE `data_vendor` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(64) NOT NULL,
  `website_url` varchar(255) DEFAULT NULL,
  `support_email` varchar(255) DEFAULT NULL,
  `created_date` datetime NOT NULL,
  `last_updated_date` datetime NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `exchange` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `abbrev` varchar(32) NOT NULL,
  `name` varchar(255) NOT NULL,
  `city` varchar(255) DEFAULT NULL,
  `country` varchar(255) DEFAULT NULL,
  `currency` varchar(64) DEFAULT NULL,
  `timezone_offset` time DEFAULT NULL,
  `created_date` datetime NOT NULL,
  `last_updated_date` datetime NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `symbol` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `exchange_id` varchar(32) NOT NULL,
  `ticker` varchar(32) DEFAULT NULL,
  `instrument` varchar(64) DEFAULT NULL,
  `name` varchar(255) DEFAULT NULL,
  `sector` varchar(255) DEFAULT NULL,
  `currency` varchar(32) DEFAULT NULL,
  `created_date` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `last_updated_date` datetime DEFAULT NULL,
  `weight` float DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `index_exchange_id` (`exchange_id`)
) ENGINE=InnoDB AUTO_INCREMENT=0 DEFAULT CHARSET=utf8;
