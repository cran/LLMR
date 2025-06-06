## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = FALSE
)

## ----setup--------------------------------------------------------------------
# library(LLMR)

## ----data---------------------------------------------------------------------
# text_input <- c(
#   Washington = "Among the vicissitudes incident to life no event could have filled me with greater anxieties than that of which the notification was transmitted by your order, and received on the 14th day of the present month. On the one hand, I was summoned by my Country, whose voice I can never hear but with veneration and love, from a retreat which I had chosen with the fondest predilection, and, in my flattering hopes, with an immutable decision, as the asylum of my declining years--a retreat which was rendered every day more necessary as well as more dear to me by the addition of habit to inclination, and of frequent interruptions in my health to the gradual waste committed on it by time. On the other hand, the magnitude and difficulty of the trust to which the voice of my country called me, being sufficient to awaken in the wisest and most experienced of her citizens a distrustful scrutiny into his qualifications, could not but overwhelm with despondence one who (inheriting inferior endowments from nature and unpracticed in the duties of civil administration) ought to be peculiarly conscious of his own deficiencies. In this conflict of emotions all I dare aver is that it has been my faithful study to collect my duty from a just appreciation of every circumstance by which it might be affected. All I dare hope is that if, in executing this task, I have been too much swayed by a grateful remembrance of former instances, or by an affectionate sensibility to this transcendent proof of the confidence of my fellow-citizens, and have thence too little consulted my incapacity as well as disinclination for the weighty and untried cares before me, my error will be palliated by the motives which mislead me, and its consequences be judged by my country with some share of the partiality in which they originated.",
#   Adams = "When it was first perceived, in early times, that no middle course for America remained between unlimited submission to a foreign legislature and a total independence of its claims, men of reflection were less apprehensive of danger from the formidable power of fleets and armies they must determine to resist than from those contests and dissensions which would certainly arise concerning the forms of government to be instituted over the whole and over the parts of this extensive country. Relying, however, on the purity of their intentions, the justice of their cause, and the integrity and intelligence of the people, under an overruling Providence which had so signally protected this country from the first, the representatives of this nation, then consisting of little more than half its present number, not only broke to pieces the chains which were forging and the rod of iron that was lifted up, but frankly cut asunder the ties which had bound them, and launched into an ocean of uncertainty.",
#   Jefferson = "Called upon to undertake the duties of the first executive office of our country, I avail myself of the presence of that portion of my fellow-citizens which is here assembled to express my grateful thanks for the favor with which they have been pleased to look toward me, to declare a sincere consciousness that the task is above my talents, and that I approach it with those anxious and awful presentiments which the greatness of the charge and the weakness of my powers so justly inspire. A rising nation, spread over a wide and fruitful land, traversing all the seas with the rich productions of their industry, engaged in commerce with nations who feel power and forget right, advancing rapidly to destinies beyond the reach of mortal eye -- when I contemplate these transcendent objects, and see the honor, the happiness, and the hopes of this beloved country committed to the issue and the auspices of this day, I shrink from the contemplation, and humble myself before the magnitude of the undertaking. Utterly, indeed, should I despair did not the presence of many whom I here see remind me that in the other high authorities provided by our Constitution I shall find resources of wisdom, of virtue, and of zeal on which to rely under all difficulties. To you, then, gentlemen, who are charged with the sovereign functions of legislation, and to those associated with you, I look with encouragement for that guidance and support which may enable us to steer with safety the vessel in which we are all embarked amidst the conflicting elements of a troubled world.",
#   Madison = "Unwilling to depart from examples of the most revered authority, I avail myself of the occasion now presented to express the profound impression made on me by the call of my country to the station to the duties of which I am about to pledge myself by the most solemn of sanctions. So distinguished a mark of confidence, proceeding from the deliberate and tranquil suffrage of a free and virtuous nation, would under any circumstances have commanded my gratitude and devotion, as well as filled me with an awful sense of the trust to be assumed. Under the various circumstances which give peculiar solemnity to the existing period, I feel that both the honor and the responsibility allotted to me are inexpressibly enhanced.",
#   Bush = "The peaceful transfer of authority is rare in history, yet common in our country. With a simple oath, we affirm old traditions and make new beginnings. As I begin, I thank President Clinton for his service to our Nation, and I thank Vice President Gore for a contest conducted with spirit and ended with grace. I am honored and humbled to stand here where so many of America's leaders have come before me, and so many will follow. We have a place, all of us, in a long story, a story we continue but whose end we will not see. It is a story of a new world that became a friend and liberator of the old, the story of a slaveholding society that became a servant of freedom, the story of a power that went into the world to protect but not possess, to defend but not to conquer.",
#   Obama = "My fellow citizens, I stand here today humbled by the task before us, grateful for the trust you have bestowed, mindful of the sacrifices borne by our ancestors. I thank President Bush for his service to our Nation, as well as the generosity and cooperation he has shown throughout this transition. Forty-four Americans have now taken the Presidential oath. The words have been spoken during rising tides of prosperity and the still waters of peace. Yet every so often, the oath is taken amidst gathering clouds and raging storms. At these moments, America has carried on not simply because of the skill or vision of those in high office, but because we the people have remained faithful to the ideals of our forebears and true to our founding documents.",
#   Trump = "We, the citizens of America, are now joined in a great national effort to rebuild our country and restore its promise for all of our people. Together, we will determine the course of America and the world for many, many years to come. We will face challenges, we will confront hardships, but we will get the job done. Every 4 years, we gather on these steps to carry out the orderly and peaceful transfer of power, and we are grateful to President Obama and First Lady Michelle Obama for their gracious aid throughout this transition. They have been magnificent. Thank you.",
#   Biden = "This is America's day. This is democracy's day, a day of history and hope, of renewal and resolve. Through a crucible for the ages America has been tested anew, and America has risen to the challenge. Today we celebrate the triumph not of a candidate, but of a cause, the cause of democracy. The people—the will of the people has been heard, and the will of the people has been heeded. We've learned again that democracy is precious, democracy is fragile. And at this hour, my friends, democracy has prevailed."
# )

## ----config-------------------------------------------------------------------
# embed_cfg <- llm_config(
#   provider = "gemini",
#   model = "embedding-001", #"text-embedding-004",
#   api_key = Sys.getenv("GEMINI_KEY"),
#   embedding = TRUE
# )

## ----embeddings---------------------------------------------------------------
# embeddings <- get_batched_embeddings(
#   texts = text_input,
#   embed_config = embed_cfg,
#   batch_size = 5
# )

## ----analysis-----------------------------------------------------------------
# # Compute correlation matrix
# cors <- cor(t(embeddings))
# print(round(cors, 2))
# 
# # Normalize embeddings for cosine similarity
# embd_normalized <- t(apply(embeddings, 1,
#                           function(x) x / sqrt(sum(x^2))))
# 
# # Compute cosine similarity matrix
# sim_matrix <- embd_normalized %*% t(embd_normalized)
# 
# # Convert similarity to distance
# dist_matrix <- 1 - sim_matrix
# 
# # Convert to a distance object
# dist_object <- as.dist(dist_matrix)
# 
# # Perform hierarchical clustering
# hc <- hclust(dist_object, method = "ward.D2")
# plot(hc, main = "Clustering of Presidential Inaugural Speeches\n(only beginning the paragraphs)")

