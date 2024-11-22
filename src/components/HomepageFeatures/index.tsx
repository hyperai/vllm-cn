import clsx from "clsx";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<"svg">>;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: "高效编程",
    Svg: require("@site/static/img/1-High-efficiency.png").default,
    description: (
      <>
        vLLM 采用创新的内存管理与执行架构，大幅提升大模型推理的速度与效率
      </>
    ),
  },
  {
    title: "实时编译",
    Svg: require("@site/static/img/2Memory-Management.png").default,
    description: (
      <>支持高度并发的请求处理，vLLM 可同时服务数千用户，提升吞吐量和响应速度</>
    ),
  },
  {
    title: "灵活的迭代空间结构",
    Svg: require("@site/static/img/3-Flexible-user-friendly.png").default,
    description: (
      <>
        兼容多种深度学习框架，vLLM 易于集成到现有机器学习管道，部署更便捷
      </>
    ),
  },
];

function Feature({ title, Svg, description }: FeatureItem) {
  return (
    <div className={clsx("col col--4")}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
