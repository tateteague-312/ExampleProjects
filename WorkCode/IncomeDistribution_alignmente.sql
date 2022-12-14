create or replace view LIV_SANDBOX.LC.HOUSEHOLDINCOME_VW(
	BU,
	UNITID,
	UNITNAME,
	LEASINGENTITYID,
	MOVEINDATE,
	LEASESTARTDATE,
	LEASEENDDATE,
	MOVEOUTDATE,
	INCOME,
	bonus,
	"Incomes Adjusted(HH)",
	BEDROOMCOUNT,
	RESIDENTCOUNT,
	roommateFlag,
	"rent",
	RENEWAL,
	RENEWED,
	DURATION,
	PMS,
	PRIORSTATE,
	STATE
) COMMENT='UPDATE: DAILY SOURCE: INTERNAL DATA'
 as 
 
--	 __    __  __    __  ______                       
--	/  |  /  |/  |  /  |/      |          __          
--	$$ |  $$ |$$ |  $$ |$$$$$$/          /  |         
--	$$ |__$$ |$$ |__$$ |  $$ |         __$$ |__       
--	$$    $$ |$$    $$ |  $$ |        /  $$    |      
--	$$$$$$$$ |$$$$$$$$ |  $$ |        $$$$$$$$/       
--	$$ |  $$ |$$ |  $$ | _$$ |_          $$ |         
--	$$ |  $$ |$$ |  $$ |/ $$   |         $$/          
--	$$/   $$/ $$/   $$/ $$$$$$/                  
-- 

--


WITH yardiLeases AS (
-- Get initial Pool of reseidents 
	SELECT DISTINCT 
		l.leasingentityid
	FROM DNA_SDW_PD.LIV_ODS.LEASE l	
	LEFT JOIN DNA_SDW_PD.LIV_ODS.UNIT u  ON l.unitid = u.unitid   
	LEFT JOIN  DNA_SDW_PD.LIV_ODS.PROPERTYMAPPING pm ON u.propertyassetid = pm.propertyassetid
	WHERE   pm.unitcount > 0
		AND pm.livcorbupropertyidentifier IS NOT NULL
		AND l.leasestartdate < current_date()
		AND l.leasestartdate IS NOT NULL 
		AND l.LEASEENDDATE  IS NOT NULL 
		AND l.leaseenddate > l.LEASESTARTDATE
		AND l.sourcetable = 'ResTenants'
		AND l.LEASESTATUS IN ('0','4','1')
		AND l.SOURCE NOT in (SELECT DISTINCT Source FROM DNA_SDW_PD.LIV_ODS.UNITEVENT WHERE DL_IsCurrent = 1 AND ActiveInd = 1 AND UnitEventName = 'Date_unit_made_ready')
	GROUP BY
		l.leasingentityid, l.source 
) 

,yardiTransition AS (
	SELECT
		leasingentityid
		,leasingentitybridgeid
		,newleid
	FROM (
		SELECT 
			yl.leasingentityid
			,ble.leasingentitybridgeid
			,ROW_NUMBER() OVER (PARTITION BY CASE WHEN ble.leasingentitybridgeid IS NULL THEN yl.leasingentityid ELSE ble.leasingentitybridgeid end  ORDER BY le.VALIDFROM DESC) AS newLEID
		FROM yardileases yl
		LEFT JOIN DNA_SDW_PD.LIV_ODS.LEASINGENTITY le ON yl.leasingentityid =  le.LEASINGENTITYID AND le.ACTIVEIND = 1
		LEFT JOIN DNA_SDW_PD.LIV_ODS.BRIDGELEASINGENTITY ble ON (lower(le."SOURCE") = lower(ble.SOURCE) AND le.SOURCEKEY = ble.sourcekey)
	)
	WHERE newLEID = 1
)
,yardiprospectids AS (
	-- Get the related LEID's from the relationship table and ones not in the relationship table also remove roomate promotino scenarios
	SELECT
		leasingentityid AS leid 
		,CASE  
			WHEN leaseholderprospectid IS NULL THEN leasingentityid
			ELSE leaseholderprospectid
		END AS leaseholderprospectid
		,CASE 
			WHEN RELATEDLEID IS NULL THEN leasingentityid
			ELSE RELATEDLEID
		END AS RELATEDLEID
		,LEASINGRELATIONSHIPVALUE
	FROM (
			SELECT 
				res.leasingentityid 
				,lr.relatedleid AS leaseholderprospectid
				,CASE 
					WHEN lr2.RELATEDLEID = res.leasingentityid THEN lr.relatedleid
					ELSE lr2.RELATEDLEID
				END AS relatedleid
				,lr2.LEASINGRELATIONSHIPVALUE 
			FROM yardiLeases res
			LEFT JOIN DNA_SDW_PD.LIV_ODS.LEASINGENTITYRELTN lr ON res.leasingentityid = lr.PRIMARYLEID AND lr.DL_ISCURRENT = 1 AND lr.activeind = 1 
			LEFT JOIN DNA_SDW_PD.LIV_ODS.LEASINGENTITYRELTN lr2 ON lr.RELATEDLEID  = lr2.PRIMARYLEID AND lr2.DL_ISCURRENT = 1 AND lr2.activeind = 1
	)
)

,OSprospectids AS(

	SELECT DISTINCT
	        L.LEASINGENTITYID AS leid
	        ,LE4.LEASINGENTITYID AS relatedleid
			,le2r.leasingrelationshipvalue
	FROM DNA_SDW_PD.LIV_ODS.LEASE L
	LEFT JOIN DNA_SDW_PD.LIV_ODS.UNIT U ON L.UNITID        = U.UNITID AND U.ACTIVEIND = 1
	LEFT JOIN DNA_SDW_PD.LIV_ODS.PROPERTYASSET P ON P.PROPERTYASSETID = U.PROPERTYASSETID AND P.ACTIVEIND   = 1
	LEFT JOIN (
			SELECT
				LEASINGENTITYID
	        	,LEASINGENTITYSOURCE
	        	,SOURCEKEY
	        	,SPLIT_PART(SOURCEKEY, '*', 4) AS LENAME
	       FROM  DNA_SDW_PD.LIV_ODS.LEASINGENTITY
	       WHERE ACTIVEIND = 1
		) LE ON  L.LEASINGENTITYID = LE.LEASINGENTITYID
	LEFT JOIN(
			SELECT
		         LEASINGENTITYID
	             ,HOUSEHOLDNAME
	             ,CONCAT(SPLIT_PART(SOURCEKEY, '*', 1), '*', SPLIT_PART(SOURCEKEY, '*', 2)) AS PROPERTY
	        FROM DNA_SDW_PD.LIV_ODS.LEASINGENTITYNAMES N
	        WHERE HOUSEHOLDNAME IS NOT NULL
	        AND ACTIVEIND  = 1
		)N ON N.HOUSEHOLDNAME = LE.LENAME AND N.PROPERTY  = P.SOURCEKEY
	
	LEFT JOIN DNA_SDW_PD.LIV_ODS.LEASINGENTITY LE2 ON N.LEASINGENTITYID = LE2.LEASINGENTITYID AND LE2.ACTIVEIND = 1
	JOIN  DNA_SDW_PD.LIV_ODS.LEASINGENTITYRELTN LE2R ON LE2.LEASINGENTITYID = LE2R.RELATEDLEID AND LE2R.ACTIVEIND = 1 AND LE2R.LEASINGRELATIONSHIPTYPE = 'Household-Contact'
	LEFT JOIN DNA_SDW_PD.LIV_ODS.LEASINGENTITY LE3 ON LE2R.PRIMARYLEID = LE3.LEASINGENTITYID AND LE3.ACTIVEIND = 1
	LEFT JOIN DNA_SDW_PD.LIV_ODS.LEASINGENTITY LE4 ON LE2R.RELATEDLEID = LE4.LEASINGENTITYID AND LE4.ACTIVEIND = 1 
)


,allprospectids AS (

	SELECT *
	FROM Osprospectids
	
	UNION
	
	SELECT LEID
		,relatedleid
		,LEASINGRELATIONSHIPVALUE
	FROM yardiprospectids

)

, ResidentInfo AS (
	-- Gather and aggregate Resident info at a Household level 
	SELECT
		pl.LEID
		,sum(pl.income) AS income
		,sum(pl.employmentadditionalincome) AS bonus
		,COUNT(pl.RELATEDLEID) AS ResidentCount
		,sum(roommateflag) roommateflag
	FROM(
		SELECT
			a2.LEID
			,a2.relatedleid
			,le.income
			,le.employmentadditionalincome
			,CASE 
				WHEN LEASINGRELATIONSHIPVALUE ilike '%room%' THEN 1
				ELSE 0
			END AS roommateFlag		
		FROM allprospectids a2
		LEFT JOIN DNA_SDW_PD.LIV_ODS.LEASINGENTITY le ON (a2.RELATEDLEID = le.leasingentityid and le.activeind = 1 and le.dl_iscurrent = 1)
		) pl
	GROUP BY pl.LEID
)

,firstLeaseInfo AS (
	--Get the first lease from multiple & remove duplicates
	SELECT *
	FROM (
		SELECT
			i.* ,
			l.* ,
			ROW_NUMBER() OVER (PARTITION BY i.LEID ORDER BY l.leasestartdate ASC ) AS dupLease
		FROM ResidentInfo i
		LEFT JOIN DNA_SDW_PD.LIV_ODS.LEASE l ON (i.LEID = l.leasingentityid AND (lower(leasestatus)IN ('current resident', 'former resident','0','4','1')))
		)
	WHERE dupLease = 1
)


,rentRate AS (
	-- Get max rent rate from rate table incasse no rent in RR and no rent on lease
	SELECT *
	FROM (
		SELECT 
			leaseid
			,leaserateamount
			,ROW_NUMBER() OVER (PARTITION BY LEASEID ORDER BY RATESTARTDATE) AS earliestRate
		FROM DNA_SDW_PD.LIV_ODS.LEASERATE lr
		WHERE lower(LEASERATETYPENAME) IN (
			'comrent'
			,'laprent'
			,'rentres'
			,'s8rent'
			,'rentpro'
			,'resrent'
			,'rentstbm'
			,'prefrent'
			,'rentstab'
			,'.rent'
			,'rent'
			,'leaserent'
			,'cha-rent'
			,'haprent'
			,'prorent'
			,'rentc'
			,'rnta'
			,'rnta2')
	)
	WHERE earliestRate = 1 
)

,propertyinfo AS (
	-- aggregate information and add in asset details
	SELECT *
	FROM (
		SELECT DISTINCT 
			a.leid as leasingentityid 
			,a.leaseid 
			,pm.livcorbupropertyidentifier AS BU 
			,pm.propertyname 
			,pm.state
			,u.unitname 
			,u.unitid
			,u.bedroomcount
			,u.bathroomcount
			,a.moveindate
			,a.leasestartdate
			,a.leaseenddate
			,a.moveoutdate 
			,a.renewallease
			,a.income 
			,a.bonus
			,a.ResidentCount
			,u.source
			,CASE 
				WHEN a.roommateflag > 0 THEN 1
				ELSE 0
			END AS roommateflag
			,a.ALTERNATIVERENT AS rent
			,rr.leaserateamount
			,pm.validto
			,ROW_NUMBER() OVER (PARTITION BY pm.livcorbupropertyidentifier,u.unitname ,a.leasestartdate, a.leasingentityid ORDER BY pm.validto DESC, a.income ASC ) AS rank_
		FROM firstLeaseInfo a
		LEFT JOIN (SELECT DISTINCT unitid,unitname, bedroomcount,bathroomcount,propertyassetid,"SOURCE" FROM DNA_SDW_PD.LIV_ODS.UNIT) u ON a.unitid = u.unitid
		LEFT JOIN (SELECT DISTINCT livcorbupropertyidentifier, propertyname, validto, propertyassetid, state FROM DNA_SDW_PD.LIV_ODS.PROPERTYMAPPING) PM ON u.propertyassetid = pm.propertyassetid
		LEFT JOIN rentRate rr ON a.leaseid = rr.leaseid
	)
	WHERE rank_ = 1
) 

, ysrentroll  AS (
	--Map in YS effective rent rates from rentroll table because we currently are unable to reference an internal rent roll or get true rent metrics
	SELECT
		DISTINCT
		    concat(sm.BU
		    	,CASE WHEN rr.building = 'N/A' THEN '' ELSE rr.building END 
		    	,rr.UNIT_NUMBER
		    	,rr.LEASE_START_DATE
		    ) AS residentkey
			,rr.effective_rent
			,rr.RENEWAL 
	FROM "DNA_SDW_PD"."LIV_YIELDSTAR"."RENTROLL" rr
	LEFT JOIN LIV_SANDBOX.MAPPING.REALPAGESUPERMAPPING sm ON lower(rr.COMMUNITY_NAME)  = lower(sm.PROPERTYNAME)	
)

, combineYS AS (
	--Join Rents & Renewal info from YS RentRoll
	SELECT *
	from (
		SELECT  
			pi.BU
			,pi.unitid
			,pi.unitname
			,pi.leasingentityid
			,pi.moveindate
			,pi.leasestartdate
			,pi.leaseenddate
			,pi.income
			,pi.moveoutdate
			,pi.bonus
			,pi.bedroomcount
			,pi.rent
			,pi.leaserateamount
			,pi.residentcount
			,pi.roommateflag
			,pi.SOURCE
			,pi.state
			,ys.effective_rent
			,COALESCE (ys.renewal, pi.renewallease) AS renewal
			,ROW_NUMBER() OVER (PARTITION BY pi.leasingentityid ORDER BY ys.effective_rent DESC) AS rentRank
		FROM propertyinfo pi
		LEFT JOIN ysrentroll ys ON concat(pi.BU, pi.unitname, pi.leasestartdate, pi.leaseenddate) = ys.residentkey
		WHERE pi.BU IS NOT NULL
	)
	WHERE rentRank = 1
)

, renewals_Durations AS (
-- Renewed: Get information to find whether they renewed and duration of stay
	SELECT leasingentityid, max(leasestartdate) AS LEASESTARTDATE, max(leaseenddate) AS LEASEENDDATE, min(moveindate) AS moveindate, max(moveoutdate) AS moveoutdate
	FROM (
		SELECT *
		FROM (
			SELECT leasingentityid, leasestartdate, leaseenddate, moveindate, moveoutdate
				,CASE WHEN (lower(leasestatus)IN ('current resident', 'former resident','0','4','1') AND leasestartdate < current_date()) THEN 1
					 ELSE 0
				END AS invalidFlag
			FROM DNA_SDW_PD.LIV_ODS.LEASE l
			)
		WHERE invalidFlag = 1 
		)
	GROUP BY leasingentityid
)


,prioraddress AS (
	SELECT 
		leasingentityid
		,priorstate
	FROM (
		SELECT DISTINCT 
			leasingentityid
			,NULLIF(trim(state),'') AS priorstate
			,validfrom
			,ROW_NUMBER() OVER (PARTITION BY LEASINGENTITYID ORDER BY VALIDFROM desc) AS duprecords
		FROM DNA_SDW_PD.LIV_ODS.ADDRESS 
		WHERE ADDRESSTYPE = 'PriorResidence'
	)
	WHERE duprecords = 1
	AND priorstate IS NOT NULL 
)

,prioraddress_yardi AS (
	SELECT 
		leasingentityid 
		,priorstate
	FROM (
		SELECT DISTINCT 
			leasingentityid
			,NULLIF(trim(PRIORSTATE),'') AS priorstate 
			,validfrom
			,ROW_NUMBER() OVER (PARTITION BY LEASINGENTITYID ORDER BY VALIDFROM desc) AS duprecords
		FROM DNA_SDW_PD.LIV_ODS.LEASINGENTITY  
	) 	
	WHERE duprecords =1
	AND priorstate IS NOT NULL 
)

,prioraddress_leid AS (
	SELECT 
		relatedleid 
		,priorstate
		,leasingentityid
	FROM (
		SELECT DISTINCT  
			tmp.*
			,ler.RELATEDLEID
			,row_number() OVER (PARTITION BY tmp.leasingentityid ORDER BY validfrom DESC) AS dupleids
		FROM prioraddress_yardi tmp
		LEFT JOIN DNA_SDW_PD.LIV_ODS.LEASINGENTITYRELTN_A ler ON tmp.leasingentityid = ler.PRIMARYLEID AND lower(LEFT(ler.RELATEDSOURCEKEY,1)) = 't'
		WHERE relatedleid IS NOT NULL 
	)
	WHERE dupleids = 1
)
-- Final Query Output

SELECT 
	BU
	,unitid
	,unitname
	,leasingentityid
	,moveindate
	,leasestartdate
	,leaseenddate
	,moveoutdate
	,income
	,bonus
	,"Incomes Adjusted(HH)"
	,bedroomcount
	,residentcount
	,roommateflag
	,"rent"
	,renewal
	,renewed
	,duration
	,pms
	,priorstate
	,state
FROM (
	SELECT 
		 td.BU
		,td.unitid 
		,td.unitname
		,td.leasingentityid
		,mx.moveindate
		,td.leasestartdate 
		,td.leaseenddate 
		,td.moveoutdate
		,td.income
		,td.bonus	
		-- Income adjustment see: https://revantage.atlassian.net/l/c/CM6rD6Ej
		, CASE 
			--WHEN td.SOURCE IN (SELECT DISTINCT Source FROM DNA_SDW_PD.LIV_ODS.UNITEVENT WHERE DL_IsCurrent = 1 AND ActiveInd = 1 AND UnitEventName = 'Date_unit_made_ready') THEN td.income + td.bonus
			WHEN td.income < 100 THEN td.income * 2000 + COALESCE(td.bonus,0)
			WHEN td.income >= 100 AND td.income <= 1.25 * COALESCE (td.effective_rent, td.rent, td.leaserateamount  ) THEN td.income * 50 + COALESCE(td.bonus,0)
			WHEN td.income > 1.25 * COALESCE (td.effective_rent, td.rent , td.leaserateamount ) AND td.income <= 1.25 * 12 * COALESCE (td.effective_rent, td.rent, td.leaserateamount  ) THEN td.income * 12 + COALESCE(td.bonus,0)
			ELSE td.income + COALESCE(td.bonus,0)
		END AS "Incomes Adjusted(HH)"
		,td.bedroomcount 
		,td.residentcount 
		,td.roommateflag
		-- If for whatever reason the unit doesn't exist in the RR then use the max rate we found and assume as effective rent
		,COALESCE (td.effective_rent, td.rent, td.leaserateamount ) AS "rent"
		,CASE 
			WHEN td.renewal IS NULL THEN (CASE 
											WHEN abs(datediff(DAY,mx.moveindate,td.leasestartdate)) > 7 THEN 'Y'
											ELSE 'N'
										END)
			ELSE td.renewal 
		END AS renewal
	
		-- Figure out if they renewed or not
		,CASE 
			WHEN mx.leasestartdate > td.leasestartdate THEN 1
			ELSE 0
		END AS renewed 
		-- Calculate duration stayed at residence
		,datediff(MONTH, mx.moveindate, COALESCE(mx.moveoutdate, current_date())) AS duration
		,ROW_NUMBER() OVER (PARTITION BY td.BU, td.UNITNAME,td.leasestartdate ORDER BY td.leasestartdate) AS dup
		,CASE
               WHEN LOWER(Source) IN ('alliance', 'avenue5', 'beam', 'bell', 'bh',
                                                'cityview', 'conam', 'fpimgmt', 'greystaryardi', 'holland',
                                                'mg', 'pbbell', 'peak', 'pinnacle', 'rangewater',
                                                'carroll', 'davlyn', 'fogelman', 'gables', 'lincoln', 'bridge','waterton','fpi') THEN 'Yardi'
               WHEN LOWER(Source) IN ('acg','carterhaston','cortland','dayrise','goldoller',
                                                'olympus','securityproperties','truamerica','westcorp',
                                                'bainbridge','cresmanagement','capreit','greystar','goldoller2','truamerica') THEN 'Onesite'
               WHEN LOWER(Source) IN ('dolben','imt','beamedhmri') THEN 'MRI'
        END AS PMS
		,COALESCE(a.priorstate,aleid.priorstate) AS priorstate
		,td.state
	FROM combineYS td
	LEFT JOIN renewals_Durations mx ON td.leasingentityid = mx.leasingentityid
	LEFT JOIN prioraddress a ON td.leasingentityid = a.LEASINGENTITYID 
	LEFT JOIN prioraddress_leid aleid ON td.leasingentityid = aleid.relatedleid
	ORDER BY td.BU,td.UNITNAME, td.LEASESTARTDATE
)
WHERE dup = 1
ORDER BY BU, UNITNAME, LEASESTARTDATE;


grant select on all views in schema LIV_SANDBOX.LC to role LIV_READONLY;
grant select on all views in schema LIV_SANDBOX.RP to role LIV_READONLY;
